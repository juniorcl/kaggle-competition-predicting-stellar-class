import os
import pandas as pd
from src.layer_stacking.config import MODEL_REGISTRY


MODEL_DIR = "models/layer_2/"

os.makedirs(MODEL_DIR, exist_ok=True)

X_TRAIN = pd.read_parquet("data/X_train_stacking_layer_one.parquet")
Y_TRAIN = pd.read_parquet("data/y_train.parquet")
Y_TRAIN_ENC = Y_TRAIN.loc[:, 'class_encoded']


if __name__ == "__main__":

    for model_name, model_instance in MODEL_REGISTRY.items():
        print(f"\n---------- Train {model_name} ----------")

        model_path = os.path.join(MODEL_DIR, f"model_{model_name}.pkl")

        if os.path.exists(model_path):
            print(f"Skipping {model_name} (already trained).")
            continue

        print(f"Training {model_name}...")
        model_instance(X_TRAIN, Y_TRAIN_ENC, model_path)
