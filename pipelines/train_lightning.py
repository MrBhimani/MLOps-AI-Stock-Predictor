import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from models.lightning_model import StockPredictor


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Close"]).values
    y = df["Close"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return TensorDataset(X, y), X.shape[1]


def main():
    dataset, input_dim = load_data("data/processed/AAPL_processed.csv")

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = StockPredictor(input_dim=input_dim)

    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
