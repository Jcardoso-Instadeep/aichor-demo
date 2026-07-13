"""metrics_test: deliberately drive CPU / memory / disk / GPU load inside an
AIchor experiment pod so the infrastructure metrics pipeline (cAdvisor,
kube-state-metrics, DCGM) has something real to scrape.

The point is to produce a clear, sustained signal on the per-experiment metrics
that Ygritte serves (aichor_experiment_cpu:*, aichor_experiment_memory:*,
aichor_experiment_disk:*, aichor_experiment_gpu:*). Run it long enough (minutes)
that several scrape intervals AND at least a few recording-rule evaluations land
while the load is on; a run that lives for only a scrape or two records almost
nothing.

Call generate_metrics() from your entrypoint (see main.py).
"""

from metrics_test.load import generate_metrics

__all__ = ["generate_metrics"]
