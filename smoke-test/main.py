import argparse
import time

from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.pytorch import pytorchop
from src.operators.xgboost import xgboostop
from src.operators.jobset import jobsetop
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


OPERATOR_TABLE = {
    "ray": rayop,
    "kuberay": rayop,
    "jax": jaxop,
    "pytorch": pytorchop,
    "xgboost": xgboostop,
    "jobset": jobsetop
}

def print_numbered_lines(target_mb=10):
    target_bytes = int(target_mb * 1024 * 1024)
    bytes_written = 0
    line_nr = 1

    n = 50
    # while bytes_written < target_bytes:
    while line_nr < n:
        if line_nr % 3000 == 0:
            time.sleep(0.5)
        # Construct the string
        output = f"This is line {line_nr}"
        
        # print() adds a newline (\n), so we add 1 to the byte count
        # On Windows, you might want to add 2 if using \r\n
        bytes_written += len(output.encode('utf-8')) + 1
        bytes_written += 1
        
        
        logging.info(output)
        line_nr += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Smoke test on any operator')
    parser.add_argument("--operator", type=str, default="jobset", choices=OPERATOR_TABLE.keys(),help="operator name")
    parser.add_argument("--sleep", type=int, default="0", help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")

    args = parser.parse_args()

    print(f"using {args.operator} operator")
    OPERATOR_TABLE[args.operator](args.tb_write)

    print_numbered_lines(5)

    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)