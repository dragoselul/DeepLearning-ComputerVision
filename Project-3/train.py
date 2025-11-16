import torch
import torch.nn.functional as F


def train(model, train_loader, loss_fn, opt, device, save_name, epochs=20,
          val_loader=None, patience=5, min_delta=0.001):
    """
    Train model with early stopping support

    Args:
        model: Model to train
        train_loader: Training data loader
        loss_fn: Loss function
        opt: Optimizer
        device: Device to train on
        save_name: Path to save model
        epochs: Maximum number of epochs
        val_loader: Validation data loader (optional, for early stopping)
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    """
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        print(f'* Epoch {epoch+1}/{epochs}')

        avg_loss = 0
        y_pred = None  # Initialize for monitoring
        y_true = None
        for X_batch, y_true in train_loader:
            X_batch = X_batch.to(device)
            y_true = y_true.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            opt.step()

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)

        print(f' - train_loss: {avg_loss:.4f}')

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    val_pred = model(X_val)
                    val_loss += loss_fn(val_pred, y_val) / len(val_loader)

            print(f' - val_loss: {val_loss:.4f}')

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                # Save best model
                torch.save(model, save_name)
                print(f'   ✓ New best model saved (val_loss: {val_loss:.4f})')
            else:
                patience_counter += 1
                print(f'   No improvement ({patience_counter}/{patience})')

                if patience_counter >= patience:
                    print(f'\n⚠ Early stopping triggered! Best epoch: {best_epoch}')
                    print(f'   Best val_loss: {best_val_loss:.4f}')
                    return save_name
        else:
            # No validation set, save model every epoch
            torch.save(model, save_name)

        # Monitor predictions to detect issues
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                if y_pred is not None:
                    sample_pred = torch.sigmoid(y_pred[0, 0]).cpu()
                    pred_binary = (sample_pred > 0.5).float()
                    white_pct = pred_binary.mean().item() * 100
                    true_white_pct = y_true[0, 0].cpu().mean().item() * 100
                    print(f'   Pred mean: {sample_pred.mean():.3f} | Pred white: {white_pct:.1f}% | True white: {true_white_pct:.1f}%')
            model.train()

    # Training completed without early stopping
    if val_loader is None:
        torch.save(model, save_name)

    print("\n✓ Training has finished!")
    if val_loader is not None:
        print(f"  Best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    return save_name
