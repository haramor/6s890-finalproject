"""
Training Script - FIXED VERSION
Addresses NaN/gradient explosion issues
"""

import sys
import os
import time
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
import h5py
from tqdm import tqdm

# Add chess-transformers to path
chess_transformers_path = Path(__file__).parent.parent.parent / "chess-transformers"
sys.path.insert(0, str(chess_transformers_path))

try:
    from chess_transformers.transformers.models import ChessTransformerEncoder
    from chess_transformers.train.datasets import ChessDataset
    from chess_transformers.train.utils import (
        AverageMeter, topk_accuracy, get_lr, change_lr, save_checkpoint
    )
except ImportError as e:
    print(f"Warning: Could not import from chess-transformers: {e}")
    print("Some functionality may be limited")

# Import our custom losses
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))
from models.entropy_loss import EntropyRegularizedLoss, LabelSmoothedCE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def create_model(config):
    """Create model from configuration."""
    try:
        model = ChessTransformerEncoder(config)
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        raise


def check_for_nan(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name}")
        return True
    return False


def train_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    config,
    writer,
    step,
    use_entropy
):
    """Train for one epoch with NaN checking."""
    model.train()

    losses = AverageMeter()
    ce_losses = AverageMeter()
    entropy_vals = AverageMeter()
    top1_acc = AverageMeter()
    top3_acc = AverageMeter()

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(DEVICE)

        # Forward pass with mixed precision
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
            predicted_moves = model(batch)

            # Check for NaN in predictions
            if check_for_nan(predicted_moves, "predictions"):
                print(f"⚠️ Skipping batch {i} due to NaN in predictions")
                continue

            # Compute loss
            if use_entropy:
                loss, ce_loss, entropy = criterion(
                    predicted=predicted_moves,
                    targets=batch["moves"][:, 1:],
                    lengths=batch["lengths"]
                )
                
                # Check for NaN in loss components
                if check_for_nan(loss, "total loss") or check_for_nan(ce_loss, "CE loss"):
                    print(f"⚠️ Skipping batch {i} due to NaN in loss")
                    continue
                    
                ce_losses.update(ce_loss.item(), batch["lengths"].sum().item())
                entropy_vals.update(entropy.item(), batch["lengths"].sum().item())
            else:
                loss = criterion(
                    predicted=predicted_moves,
                    targets=batch["moves"][:, 1:],
                    lengths=batch["lengths"]
                )
                
                if check_for_nan(loss, "loss"):
                    print(f"⚠️ Skipping batch {i} due to NaN in loss")
                    continue
                    
                ce_loss = loss

            loss = loss / config.BATCHES_PER_STEP

        # Backward pass
        scaler.scale(loss).backward()

        # Check for NaN in gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_for_nan(param.grad, f"gradient {name}"):
                    has_nan_grad = True
                    break
        
        if has_nan_grad:
            print(f"⚠️ NaN in gradients at step {step}, zeroing gradients and skipping update")
            optimizer.zero_grad()
            continue

        # Track metrics
        losses.update(loss.item() * config.BATCHES_PER_STEP, batch["lengths"].sum().item())

        # Compute accuracy
        try:
            top1, top3, _ = topk_accuracy(
                logits=predicted_moves[:, 0, :],
                targets=batch["moves"][:, 1],
                k=[1, 3, 5]
            )
            top1_acc.update(top1, batch["lengths"].shape[0])
            top3_acc.update(top3, batch["lengths"].shape[0])
        except:
            pass

        # Update weights every BATCHES_PER_STEP
        if (i + 1) % config.BATCHES_PER_STEP == 0:
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1

            # Update learning rate
            try:
                new_lr = get_lr(
                    step=step,
                    d_model=config.D_MODEL,
                    warmup_steps=config.WARMUP_STEPS,
                    schedule=config.LR_SCHEDULE,
                    decay=config.LR_DECAY
                )
                change_lr(optimizer, new_lr)
            except:
                pass

            # Log to tensorboard
            if step % config.PRINT_FREQUENCY == 0:
                writer.add_scalar("train/loss", losses.val, step)
                writer.add_scalar("train/ce_loss", ce_losses.val if use_entropy else losses.val, step)
                if use_entropy:
                    writer.add_scalar("train/entropy", entropy_vals.val, step)
                writer.add_scalar("train/top1_acc", top1_acc.val, step)
                writer.add_scalar("train/top3_acc", top3_acc.val, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

                if use_entropy:
                    print(f"Step {step}/{config.N_STEPS} | "
                          f"Loss: {losses.val:.4f} | "
                          f"CE: {ce_losses.val:.4f} | "
                          f"Entropy: {entropy_vals.val:.4f} | "
                          f"Top1: {top1_acc.val:.3f} | "
                          f"Top3: {top3_acc.val:.3f}")
                else:
                    print(f"Step {step}/{config.N_STEPS} | "
                          f"Loss: {losses.val:.4f} | "
                          f"Top1: {top1_acc.val:.3f} | "
                          f"Top3: {top3_acc.val:.3f}")

            # Stop if we've reached target steps
            if step >= config.N_STEPS:
                return step

    return step


def validate(val_loader, model, criterion, config, writer, epoch, use_entropy):
    """Validate the model."""
    model.eval()

    losses = AverageMeter()
    ce_losses = AverageMeter()
    entropy_vals = AverageMeter()
    top1_acc = AverageMeter()
    top3_acc = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
                predicted_moves = model(batch)
                
                # Skip if NaN
                if check_for_nan(predicted_moves, "val predictions"):
                    continue

                # Compute loss
                if use_entropy:
                    loss, ce_loss, entropy = criterion(
                        predicted_moves,
                        batch["moves"][:, 1:],
                        batch["lengths"]
                    )
                    
                    if not (torch.isnan(ce_loss) or torch.isnan(entropy)):
                        ce_losses.update(ce_loss.item(), batch["lengths"].sum().item())
                        entropy_vals.update(entropy.item(), batch["lengths"].sum().item())
                else:
                    loss = criterion(
                        predicted_moves,
                        batch["moves"][:, 1:],
                        batch["lengths"]
                    )
                
                if not torch.isnan(loss):
                    losses.update(loss.item(), batch["lengths"].sum().item())

            # Compute accuracy
            try:
                top1, top3, _ = topk_accuracy(
                    logits=predicted_moves[:, 0, :],
                    targets=batch["moves"][:, 1],
                    k=[1, 3, 5]
                )
                top1_acc.update(top1, batch["lengths"].shape[0])
                top3_acc.update(top3, batch["lengths"].shape[0])
            except:
                pass

    # Log validation metrics
    writer.add_scalar("val/loss", losses.avg, epoch)
    writer.add_scalar("val/ce_loss", ce_losses.avg if use_entropy else losses.avg, epoch)
    if use_entropy:
        writer.add_scalar("val/entropy", entropy_vals.avg, epoch)
    writer.add_scalar("val/top1_acc", top1_acc.avg, epoch)
    writer.add_scalar("val/top3_acc", top3_acc.avg, epoch)

    if use_entropy:
        print(f"\nValidation | Loss: {losses.avg:.4f} | CE: {ce_losses.avg:.4f} | "
              f"Entropy: {entropy_vals.avg:.4f} | Top1: {top1_acc.avg:.3f} | "
              f"Top3: {top3_acc.avg:.3f}\n")
    else:
        print(f"\nValidation | Loss: {losses.avg:.4f} | "
              f"Top1: {top1_acc.avg:.3f} | Top3: {top3_acc.avg:.3f}\n")

    return losses.avg, top1_acc.avg


def main():
    parser = argparse.ArgumentParser(description="Train chess behavioral cloning model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Create directories
    Path(config.CHECKPOINT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(config.LOGS_FOLDER).mkdir(parents=True, exist_ok=True)

    # Initialize tensorboard
    writer = SummaryWriter(log_dir=config.LOGS_FOLDER)

    print(f"\nExperiment: {config.NAME}")
    print(f"Type: {config.EXPERIMENT_TYPE}")
    print(f"Device: {DEVICE}")
    
    # Check which regularization to use
    use_entropy = getattr(config, 'USE_ENTROPY_REGULARIZATION', False)
    
    if use_entropy:
        print(f"Entropy Regularization: True")
        print(f"Entropy Weight: {config.ENTROPY_WEIGHT}")
    else:
        print("Standard cross-entropy training")
    
    print("\n⚠️  TRAINING WITH NaN PROTECTION ENABLED")
    print("   - Gradient clipping (max_norm=1.0)")
    print("   - NaN detection in forward/backward pass")
    print("   - Batch skipping on NaN detection")

    # Create dataloaders
    print("\nLoading data...")
    try:
        train_loader = DataLoader(
            ChessDataset(
                data_folder=config.DATA_FOLDER,
                h5_file=config.H5_FILE,
                split="train",
                n_moves=config.N_MOVES
            ),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=True,
            pin_memory=config.PIN_MEMORY
        )

        val_loader = DataLoader(
            ChessDataset(
                data_folder=config.DATA_FOLDER,
                h5_file=config.H5_FILE,
                split="val",
                n_moves=config.N_MOVES
            ),
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            pin_memory=config.PIN_MEMORY
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create model
    print("\nInitializing model...")
    model = create_model(config).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Create criterion
    if use_entropy:
        criterion = EntropyRegularizedLoss(
            eps=config.LABEL_SMOOTHING,
            n_predictions=config.N_MOVES,
            entropy_weight=config.ENTROPY_WEIGHT
        ).to(DEVICE)
        print("Using entropy-regularized loss")
    else:
        criterion = LabelSmoothedCE(
            eps=config.LABEL_SMOOTHING,
            n_predictions=config.N_MOVES
        ).to(DEVICE)
        print("Using standard cross-entropy loss")

    # Create optimizer with LOWER learning rate
    initial_lr = 1e-4  # Lower than default
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        betas=config.BETAS,
        eps=config.EPSILON
    )
    print(f"Initial learning rate: {initial_lr}")

    # AMP scaler
    scaler = GradScaler(device=DEVICE, enabled=config.USE_AMP)

    # Training loop
    start_epoch = 0
    step = 0
    epochs = (config.N_STEPS // (len(train_loader) // config.BATCHES_PER_STEP)) + 1

    print(f"\nTraining for up to {epochs} epochs ({config.N_STEPS} steps)")
    print("="*60)

    best_val_acc = 0.0

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)

        # Train
        step = train_epoch(
            train_loader, model, criterion, optimizer,
            scaler, epoch, config, writer, step, use_entropy
        )

        # Check if we've reached target steps
        if step >= config.N_STEPS:
            print(f"\nReached target steps ({config.N_STEPS}), stopping training.")
            
        # Validate
        if (epoch + 1) % max(1, (config.EVAL_FREQUENCY // 100)) == 0 or step >= config.N_STEPS:
            val_loss, val_acc = validate(val_loader, model, criterion, config, writer, epoch, use_entropy)

            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, Path(config.CHECKPOINT_FOLDER) / f"best_{config.NAME}.pt")
                    print(f"Saved best checkpoint (val_acc: {val_acc:.3f})")
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")

        # Save periodic checkpoint
        if (step % config.SAVE_FREQUENCY == 0 or step >= config.N_STEPS) and step > 0:
            try:
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, Path(config.CHECKPOINT_FOLDER) / f"checkpoint_step_{step}.pt")
                print(f"Saved checkpoint at step {step}")
            except Exception as e:
                print(f"Warning: Could not save periodic checkpoint: {e}")

        if step >= config.N_STEPS:
            break

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_FOLDER}")
    print(f"Logs saved to: {config.LOGS_FOLDER}")
    print("="*60)


if __name__ == "__main__":
    main()
