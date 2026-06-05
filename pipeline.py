import subprocess
import sys

from src import NUM_LAYERS


def run(cmd):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    for i in range(1, NUM_LAYERS + 1):
        layer = f"layer_{i}"
        run(f"python main.py --layer {layer}")
        if i < NUM_LAYERS:
            run(f"python main.py --stack {layer}")
    print("=== Pipeline completo ===")
