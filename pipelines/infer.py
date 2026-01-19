import torch
import pandas as pd
import argparse

from models.lightning_model import StockPredictor


def load_model(model_path: str, input_dim: int):
    model = StockPredictor(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Stock price inference")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    X = torch.tensor(
        df.drop(columns=["Close"]).values,
        dtype=torch.float32,
    )

    model = load_model(
        args.model_path,
        input_dim=X.shape[1],
    )

    with torch.no_grad():
        predictions = model(X).squeeze().numpy()

    print("Predictions:")
    print(predictions[:10])


if __name__ == "__main__":
    main()
