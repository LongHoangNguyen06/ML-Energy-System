import os
import pickle
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR as Scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from pipeline.models.dataset import TimeSeriesDataset
from pipeline.models.transformer import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training loop
def train(model, train_loader, optimizer, criterion, scheduler, hyperparameters):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for (past, forecast), targets in progress_bar:
        past, forecast, targets = (
            past.to(device),
            forecast.to(device),
            targets.to(device),
        )
        optimizer.zero_grad()
        outputs = model((past, forecast))
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), hyperparameters.train.clip_grad
        )
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    scheduler.step()  # Update the learning rate
    progress_bar.close()
    return total_loss / len(train_loader)


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for (past, forecast), targets in progress_bar:
            past, forecast, targets = (
                past.to(device),
                forecast.to(device),
                targets.to(device),
            )
            outputs = model((past, forecast))
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    progress_bar.close()
    return total_loss / len(val_loader)


def train_loop(
    hyperparameters,
    df,
    train_id,
    merge_train_val: bool = False,
    log_wandb: bool = True,
    patience: int = 10,
):
    """
    Main training loop for the TimeSeriesTransformer model.
    Args:
    hyperparameters : DotMap
        Configuration object.
    df : pd.DataFrame
        Dataframe containing the time series data.
    train_id : int
        Unique identifier for the training run.
    merge_train_val : bool
        Whether to merge the training and validation sets.
    Returns:
        best_val_loss : float
            Best validation loss achieved during training.
    """
    randomseed = 42
    random.seed(randomseed)
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.cuda.manual_seed_all(randomseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    experiment_path = os.path.join(hyperparameters.model.save_path, f"{train_id}")
    model_path = os.path.join(experiment_path, "model.pth")
    hyperparameters_path = os.path.join(experiment_path, "hyperparameters.pth")
    os.makedirs(experiment_path, exist_ok=True)
    pickle.dump(hyperparameters, open(hyperparameters_path, "wb"))
    # Initialize data
    if merge_train_val:
        train_df = df[df["train"] | df["val"]].reset_index(drop=True)
        val_df = df[df["test"]].reset_index(drop=True)
    else:
        train_df = df[df["train"]].reset_index(drop=True)
        val_df = df[df["val"]].reset_index(drop=True)

    train_dataset = TimeSeriesDataset(train_df, hyperparameters=hyperparameters)
    val_dataset = TimeSeriesDataset(val_df, hyperparameters=hyperparameters)

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters.train.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters.train.batch_size, shuffle=False
    )

    # Initialize model
    model = build_model(hyperparameters=hyperparameters)
    model = model.to(device)
    optimizer = hyperparameters.train.optimizer(
        model.parameters(), lr=hyperparameters.train.lr
    )
    criterion = hyperparameters.train.loss()

    scheduler = Scheduler(
        optimizer,
        T_max=hyperparameters.train.epochs,
        eta_min=hyperparameters.train.min_lr,
    )

    # Main training loop
    num_epochs = hyperparameters.train.epochs
    patience_counter = 0  # Initialize the patience counter
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            hyperparameters=hyperparameters,
        )
        val_loss = validate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0  # Reset patience counter on improvement
        else:
            patience_counter += 1  # Increment patience counter if no improvement

        if patience_counter >= patience:
            print(
                f"Stopping early due to no improvement in validation loss for {patience} epochs."
            )
            break  # Exit the loop if patience has run out

        if log_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "learning_rate": lr,
                }
            )
        else:
            print("wandb is not initialized, skipping log.")
    return best_val_loss
