import subprocess
import sys


def run(cmd):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    run("python main.py --layer layer_one")
    run("python main.py --stack layer_one")
    run("python main.py --layer layer_two --data data/X_train_stacking_layer_one.parquet")
    run("python main.py --stack layer_two")
    print("=== Pipeline completo ===")
