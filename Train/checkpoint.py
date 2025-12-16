import torch
import os


def save_checkpoint(model, optimizer, epoch, avg_loss, filename):
    """
    Saves model in case of training disruption and for evaluation of training integrity
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_loss": avg_loss,
    }

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    """Loads previous checkpoint in case of training failure"""

    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)

        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        avg_loss = checkpoint["avg_loss"]

        print(
            f"Resuming training from Epoch {start_epoch}, previous Loss: {avg_loss:.4f}"
        )
        return start_epoch
    else:
        print(f"No checkpoint found at {filename}. Starting training from Epoch 0.")
        return 0
