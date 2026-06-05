import os
import sys
import time
import argparse
import pandas as pd

from src import MODEL_REGISTRY, LAYERS
from src.utils.logging_setup import setup_model_logger

RESULTS_DIR = 'results'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'training_results.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_result(model_name: str, status: str, log_loss: float | str,
                best_params: dict | str, duration: float, error: str = ''):
    record = {
        'model': model_name,
        'status': status,
        'log_loss': log_loss,
        'best_params': str(best_params),
        'duration_seconds': round(duration, 1),
        'error': error,
    }

    df = pd.DataFrame([record])
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
        existing = existing[existing['model'] != model_name]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)


def train_models(models, X_train, y_train, logger, force=False):
    total_start = time.perf_counter()
    for model_cfg in models:
        name = model_cfg['name']
        model_path = model_cfg['model_path']

        if not force and os.path.exists(model_path):
            logger.info(f"Skipping {name} (exists: {model_path})")
            continue

        logger.info(f"--- Training {name} ---")
        start = time.perf_counter()
        try:
            best_value, best_params = model_cfg['func'](
                X_train, y_train,
                target='class_encoded',
                model_path=model_cfg['model_path'],
            )
            duration = time.perf_counter() - start
            logger.info(f"Completed {name} in {duration:.1f}s (loss={best_value:.4f})")
            save_result(name, 'success', best_value, best_params, duration)
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error(f"FAILED {name} after {duration:.1f}s: {e}", exc_info=True)
            save_result(name, 'failed', '', '', duration, str(e))
    total_duration = time.perf_counter() - total_start
    logger.info(f"Training done in {total_duration:.1f}s")


def generate_stack(layer: str, logger):
    from src import generate_stacking_features
    if layer == "layer_1":
        train_path, test_path = "data/X_train_raw.parquet", "data/X_test_raw.parquet"
    else:
        prev = layer.rsplit("_", 1)[0] + "_" + str(int(layer.split("_")[1]) - 1)
        train_path = f"data/X_train_stacking_{prev}.parquet"
        test_path = f"data/X_test_stacking_{prev}.parquet"

    logger.info(f"Loading train: {train_path}, test: {test_path}")
    X_train = pd.read_parquet(train_path)
    y_train = pd.read_parquet("data/y_train.parquet")
    X_test = pd.read_parquet(test_path)

    generate_stacking_features(
        X_train, y_train, X_test,
        f"models/{layer}",
        output_path_train=f"data/X_train_stacking_{layer}.parquet",
        output_path_test=f"data/X_test_stacking_{layer}.parquet",
    )


def resolve_data_path(layer: str) -> str:
    if layer == "layer_1":
        return "data/X_train_raw.parquet"
    num = int(layer.split("_")[1])
    prev = f"layer_{num - 1}"
    return f"data/X_train_stacking_{prev}.parquet"


def main():
    parser = argparse.ArgumentParser(description="Kaggle Stellar Classification Pipeline")
    parser.add_argument("--layer", help="Train models for a specific layer (e.g. layer_1, layer_2, ...)")
    parser.add_argument("--data", default=None, help="Training data parquet path (overrides auto-resolve)")
    parser.add_argument("--stack", help="Generate stacking features for a layer (e.g. layer_1, layer_2, ...)")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")
    args = parser.parse_args()

    logger = setup_model_logger('main')

    if args.stack:
        generate_stack(args.stack, logger)
        return

    if args.layer:
        if args.layer not in LAYERS:
            logger.error(f"Unknown layer: {args.layer}. Available: {list(LAYERS.keys())}")
            sys.exit(1)
        models = LAYERS[args.layer]
        data_path = args.data or resolve_data_path(args.layer)
    else:
        models = MODEL_REGISTRY
        data_path = "data/X_train_raw.parquet"

    logger.info(f"Loading data: {data_path}")
    X_train = pd.read_parquet(data_path)
    y_train = pd.read_parquet("data/y_train.parquet")
    logger.info(f"Train shape: {X_train.shape}")

    train_models(models, X_train, y_train, logger, force=args.force)
    logger.info(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
