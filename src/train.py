import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from dataset import QM9GapDataset
from model import GCNModel
from evaluate import evaluate


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
SEED = 42

MODEL_PATH = "models/gcn_gap.pt"


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0

    for batch in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        ).view(-1)

        targets = batch.y.view(-1)

        loss = criterion(predictions, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


if __name__ == "__main__":

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = QM9GapDataset()

    print(f"Total molecules: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(f"Training molecules: {len(train_dataset)}")
    print(f"Validation molecules: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = GCNModel().to(device)

    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_mse = float("inf")

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_mse, val_mae = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train MSE: {train_mse:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        if val_mse < best_val_mse:

            best_val_mse = val_mse

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                f"  -> Saved best model "
                f"(Val MSE: {val_mse:.6f})"
            )

    print("\nTraining complete!")
    print(f"Best validation MSE: {best_val_mse:.6f}")
    print(f"Model saved to: {MODEL_PATH}")