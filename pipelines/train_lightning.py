import os
import glob
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from models.lightning_model import StockPredictor


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Close"]).values
    y = df["Close"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return TensorDataset(X, y), X.shape[1]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    processed_dir = to_absolute_path(cfg.data.processed_dir)
    csv_files = glob.glob(os.path.join(processed_dir, "*_processed.csv"))

    if not csv_files:
        raise ValueError(f"No processed CSV files found in {processed_dir}")

    for csv_path in csv_files:
        print(f"Training on {os.path.basename(csv_path)}")

        dataset, input_dim = load_data(csv_path)

        train_loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
        )

        model = StockPredictor(
            input_dim=input_dim,
            lr=cfg.training.learning_rate,
        )

        mlflow.set_tracking_uri("file:./mlruns")

        mlflow_logger = MLFlowLogger(
            experiment_name="stock_prediction",
            tracking_uri="file:./mlruns",
        )
        
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            logger=mlflow_logger,
        )
        
        mlflow_logger.log_hyperparams({
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "max_epochs": cfg.training.max_epochs,
        })


        trainer.fit(model, train_loader)

        os.makedirs("artifacts/models", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/models/stock_predictor.pt")

if __name__ == "__main__":
    main()
