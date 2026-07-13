"""Sustained CPU / memory / disk / GPU load, so an AIchor experiment produces
real infrastructure metrics.

Why this exists: Ygritte (and the AIchor experiments page) reads per-experiment
metrics that are derived from what cAdvisor, kube-state-metrics and DCGM scrape
off the pods. An experiment that finishes in seconds, or that idles, records
almost nothing — the recording rules only capture what is happening while they
evaluate. This module deliberately keeps the pod busy for a fixed duration so the
CPU / memory / disk / GPU panels have a clear, non-empty signal.

Everything is best-effort and non-fatal: a missing GPU, a small disk, or a
locked-down egress path degrades to "that dimension is skipped", never a crash,
so it is safe to drop into any operator's run.

Knobs (env overrides, so you can tune from the manifest without editing code):

    METRICS_TEST_SECONDS       total run time (also settable via the caller)
    METRICS_TEST_MEM_MB        resident memory to hold          (default 512)
    METRICS_TEST_CPU_WORKERS   CPU-burn processes               (default: all cores)
    METRICS_TEST_DISK          1/0 enable disk I/O              (default 1)
    METRICS_TEST_DISK_MB       bytes rewritten each disk cycle  (default 128)
    METRICS_TEST_DISK_PATH     scratch dir                      (default /tmp)
    METRICS_TEST_GPU           1/0 enable GPU load if present   (default 0; dev has no GPUs)
    METRICS_TEST_GPU_MATRIX    square-matrix side for matmul    (default 4096)
    METRICS_TEST_NETWORK       1/0 enable egress load           (default 0)
"""

import logging
import math
import os
import socket
import threading
import time
from multiprocessing import Process

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# small env helpers
# ---------------------------------------------------------------------------

def _env_int(name, default):
    try:
        v = os.environ.get(name)
        return int(v) if v not in (None, "") else default
    except ValueError:
        logger.warning("%s=%r is not an int; using %d", name, os.environ.get(name), default)
        return default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# individual load generators — each loops until `deadline` (an epoch second)
# ---------------------------------------------------------------------------

def _cpu_worker(deadline):
    """Pin a core: a tight FP loop. Runs in its own process so N of them
    actually use N cores (and thus show up as CPU usage up to the pod's limit,
    where they get throttled — which itself surfaces on the throttling panel)."""
    x = 1.0000001
    # Batch the work between clock reads so time.time() isn't the bottleneck.
    while time.time() < deadline:
        for _ in range(200_000):
            x = math.sin(x) * math.cos(x) + 1.1
            x = math.sqrt(x * x + 1.0)


def _hold_memory(mb, deadline, touch_interval=10.0):
    """Allocate `mb` MiB and keep it *resident* (working set), which is what the
    memory panels track — a bare allocation the kernel never backs with pages
    wouldn't show. We fault every page in, then re-touch periodically so nothing
    gets reclaimed while we idle-wait for the deadline. Held in the calling
    thread so its reference stays alive for the whole run."""
    n = max(1, mb) * 1024 * 1024
    logger.info("memory: allocating and holding ~%d MiB resident", mb)
    block = bytearray(n)
    page = 4096
    for i in range(0, n, page):          # fault pages in -> counts as working set
        block[i] = 1
    while time.time() < deadline:
        for i in range(0, n, page):      # re-touch so it stays resident
            block[i] = (block[i] + 1) & 0xFF
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(touch_interval, remaining))
    return block  # keep alive until here


def _disk_worker(deadline, path, file_mb):
    """Rewrite + reread a scratch file in a loop -> disk read/write bytes and
    IOPS. fsync forces the writes out so they aren't just page-cache."""
    chunk = os.urandom(1024 * 1024)  # 1 MiB
    scratch = os.path.join(path, f"metrics_test_scratch_{os.getpid()}.bin")
    logger.info("disk: cycling %d MiB writes/reads at %s", file_mb, scratch)
    try:
        while time.time() < deadline:
            with open(scratch, "wb") as f:
                for _ in range(max(1, file_mb)):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            with open(scratch, "rb") as f:
                while f.read(1024 * 1024):
                    pass
    except OSError as e:
        logger.warning("disk load stopped early: %s", e)
    finally:
        try:
            os.remove(scratch)
        except OSError:
            pass


def _gpu_worker(deadline, size):
    """Keep the GPU busy with back-to-back matmuls (drives SM/Tensor activity)
    and hold a couple of large tensors resident (drives framebuffer-used). No-op
    if torch or a CUDA device isn't there."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - torch may be absent in some images
        logger.warning("gpu: torch unavailable, skipping GPU load (%s)", e)
        return
    if not torch.cuda.is_available():
        logger.warning("gpu: CUDA not available, skipping GPU load")
        return
    dev = torch.device("cuda:0")
    logger.info("gpu: load on %s, %dx%d matmuls", torch.cuda.get_device_name(0), size, size)
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)
    c = None
    try:
        while time.time() < deadline:
            for _ in range(20):  # a burst between clock reads keeps utilisation high
                c = a @ b
                a = c * 0.5 + 0.5
            torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - never let GPU trouble fail the run
        logger.warning("gpu load stopped early: %s", e)
    finally:
        del a, b, c
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _network_worker(deadline, urls):
    """Best-effort egress so the network panels move. Off by default because a
    locked-down experiment may have no egress; failures are swallowed."""
    import urllib.request

    logger.info("network: fetching %d URL(s) in a loop", len(urls))
    while time.time() < deadline:
        for url in urls:
            if time.time() >= deadline:
                break
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    while resp.read(65536):
                        pass
            except Exception as e:  # noqa: BLE001 - egress may be blocked
                logger.warning("network fetch failed for %s: %s", url, e)
                time.sleep(2)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def _describe_pod():
    """A one-line note of which pod/experiment this is, so the load is easy to
    line up with what you see in Ygritte."""
    fields = {
        "host": socket.gethostname(),
        "job_index": os.environ.get("JOB_GLOBAL_INDEX"),
        # AIchor exposes the experiment id under a few names across operators;
        # log whichever is present.
        "experiment_id": os.environ.get("EXPERIMENT_ID")
        or os.environ.get("AICHOR_EXPERIMENT_ID"),
        "namespace": os.environ.get("NAMESPACE") or os.environ.get("POD_NAMESPACE"),
    }
    return " ".join(f"{k}={v}" for k, v in fields.items() if v)


def generate_metrics(
    duration_s=300,
    *,
    cpu=True,
    mem=True,
    disk=None,
    gpu=None,
    network=None,
    cpu_workers=None,
    mem_mb=None,
    disk_file_mb=None,
    disk_path=None,
    network_urls=None,
):
    """Drive load for `duration_s` seconds so the pod emits real metrics.

    CPU burn runs in separate processes (to use multiple cores); disk, GPU and
    network run in background threads; memory is held in the calling thread,
    which blocks until the deadline. All optional dimensions default to their
    METRICS_TEST_* env value, so the manifest can tune them without code changes.
    Returns nothing; it simply blocks for ~duration_s.
    """
    if duration_s <= 0:
        logger.info("metrics_test: duration_s<=0, nothing to do")
        return

    # env fallbacks for the optional knobs
    disk = _env_bool("METRICS_TEST_DISK", True) if disk is None else disk
    gpu = _env_bool("METRICS_TEST_GPU", False) if gpu is None else gpu
    network = _env_bool("METRICS_TEST_NETWORK", False) if network is None else network
    mem_mb = _env_int("METRICS_TEST_MEM_MB", 512) if mem_mb is None else mem_mb
    cpu_workers = (
        _env_int("METRICS_TEST_CPU_WORKERS", os.cpu_count() or 1)
        if cpu_workers is None
        else cpu_workers
    )
    disk_file_mb = _env_int("METRICS_TEST_DISK_MB", 128) if disk_file_mb is None else disk_file_mb
    disk_path = os.environ.get("METRICS_TEST_DISK_PATH", "/tmp") if disk_path is None else disk_path
    gpu_matrix = _env_int("METRICS_TEST_GPU_MATRIX", 4096)
    if network_urls is None:
        network_urls = [
            "https://storage.googleapis.com/",
            "https://www.google.com/",
        ]

    deadline = time.time() + duration_s
    logger.info(
        "metrics_test: START %ds  cpu=%s(%d workers) mem=%s(%dMiB) disk=%s gpu=%s network=%s  [%s]",
        duration_s, cpu, cpu_workers, mem, mem_mb, disk, gpu, network, _describe_pod(),
    )

    procs = []
    if cpu:
        for _ in range(max(1, cpu_workers)):
            p = Process(target=_cpu_worker, args=(deadline,), daemon=True)
            p.start()
            procs.append(p)

    threads = []
    if disk:
        threads.append(threading.Thread(
            target=_disk_worker, args=(deadline, disk_path, disk_file_mb), daemon=True))
    if gpu:
        threads.append(threading.Thread(
            target=_gpu_worker, args=(deadline, gpu_matrix), daemon=True))
    if network:
        threads.append(threading.Thread(
            target=_network_worker, args=(deadline, network_urls), daemon=True))
    for t in threads:
        t.start()

    try:
        # Hold memory (and thereby block) for the duration; if memory is off,
        # just wait out the deadline so the other generators keep running.
        if mem:
            _hold_memory(mem_mb, deadline)
        else:
            while time.time() < deadline:
                time.sleep(min(5.0, max(0.0, deadline - time.time())))
    finally:
        # Reap the CPU processes; terminate any stragglers.
        for p in procs:
            p.join(timeout=10)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for t in threads:
            t.join(timeout=15)
        logger.info("metrics_test: DONE after ~%ds", duration_s)
