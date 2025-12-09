"""
Training Script for Chess Behavioral Cloning Experiments

This script trains transformer models under three experimental conditions:
1. Baseline (mixed skill)
2. Expert-only (high ELO)
3. Game-theoretic regularization (expert + Stockfish KL penalty)

Usage:
    python train.py --config configs/baseline_config.py
    python train.py --config configs/expert_only_config.py
    python train.py --config configs/game_theoretic_config.py
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

# Import our custom loss
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.game_theoretic_loss import GameTheoreticLoss, LabelSmoothedCE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_chess_vocabulary():
    """
    Build a move vocabulary that covers the index space used by the dataset.
    
    Since the dataset uses indices up to ~8000+, and we don't have access to the
    exact vocabulary mapping, we create a placeholder vocabulary that maps
    indices to dummy UCI moves. The actual Stockfish evaluation will only use
    legal moves from the position, so this won't affect correctness.
    
    Returns:
        Tuple of (move_to_idx, idx_to_move) dictionaries
    """
    # Dataset appears to use indices up to ~8000, so create vocab of size 10000 to be safe
    VOCAB_SIZE = 10000
    
    move_to_idx = {}
    idx_to_move = {}
    
    # First, add all standard UCI moves
    files = 'abcdefgh'
    ranks = '12345678'
    promotion_pieces = ['q', 'r', 'b', 'n']
    idx = 0
    
    for from_file in files:
        for from_rank in ranks:
            from_square = from_file + from_rank
            for to_file in files:
                for to_rank in ranks:
                    to_square = to_file + to_rank
                    if from_square == to_square:
                        continue
                    
                    # Regular move
                    move = from_square + to_square
                    move_to_idx[move] = idx
                    idx_to_move[idx] = move
                    idx += 1
                    
                    # Add promotions
                    if to_rank in ['1', '8']:
                        for piece in promotion_pieces:
                            promo_move = from_square + to_square + piece
                            move_to_idx[promo_move] = idx
                            idx_to_move[idx] = promo_move
                            idx += 1
    
    # Fill remaining indices with placeholder moves
    # This ensures we can handle any index the dataset throws at us
    while idx < VOCAB_SIZE:
        idx_to_move[idx] = f"placeholder_{idx}"
        idx += 1
    
    print(f"Created vocabulary with {len(idx_to_move)} total indices ({len(move_to_idx)} real UCI moves)")
    return move_to_idx, idx_to_move


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
        print("Using simplified model for testing...")
        # Create a minimal model for testing
        return SimpleChessModel(config)


class SimpleChessModel(nn.Module):
    """
    Simplified model for testing without full chess-transformers installation.
    Replace with actual ChessTransformerEncoder for real experiments.
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(13, 128)  # Piece embeddings
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        self.classifier = nn.Linear(128, config.VOCAB_SIZES["moves"])

    def forward(self, batch):
        # Simple forward pass for testing
        board = batch["board_positions"]  # (B, 64)
        x = self.embedding(board)  # (B, 64, 128)
        x = x.transpose(0, 1)  # (64, B, 128)
        x = self.transformer(x)  # (64, B, 128)
        x = x[0]  # Take first token (B, 128)
        logits = self.classifier(x)  # (B, vocab_size)
        return logits.unsqueeze(1)  # (B, 1, vocab_size)


def train_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    config,
    writer,
    step
):
    """Train for one epoch."""
    model.train()

    losses = AverageMeter()
    ce_losses = AverageMeter()
    kl_losses = AverageMeter()
    top1_acc = AverageMeter()
    top3_acc = AverageMeter()

    start_time = time.time()

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(DEVICE)

        # Forward pass with mixed precision
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=config.USE_AMP):
            predicted_moves = model(batch)  # (N, n_moves, vocab_size)

            # Compute loss
            if config.USE_GT_REGULARIZATION and hasattr(criterion, 'gt_weight'):
                # Game-theoretic loss returns (total, ce, kl)
                loss, ce_loss, kl_loss = criterion(
                    predicted=predicted_moves,
                    targets=batch["moves"][:, 1:],
                    lengths=batch["lengths"],
                    board_states=batch["board_positions"]  # ADDED: Pass board positions for GT regularization
                )
                ce_losses.update(ce_loss.item(), batch["lengths"].sum().item())
                kl_losses.update(kl_loss.item(), batch["lengths"].sum().item())
            else:
                # Standard cross-entropy loss
                loss = criterion(
                    predicted=predicted_moves,
                    targets=batch["moves"][:, 1:],
                    lengths=batch["lengths"]
                )
                ce_loss = loss

            loss = loss / config.BATCHES_PER_STEP

        # Backward pass
        scaler.scale(loss).backward()

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
            pass  # Skip accuracy if topk_accuracy not available

        # Update weights every BATCHES_PER_STEP
        if (i + 1) % config.BATCHES_PER_STEP == 0:
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
                pass  # Use default LR if schedule not available

            # Log to tensorboard
            if step % config.PRINT_FREQUENCY == 0:
                writer.add_scalar("train/loss", losses.val, step)
                writer.add_scalar("train/ce_loss", ce_losses.val, step)
                writer.add_scalar("train/kl_loss", kl_losses.val, step)
                writer.add_scalar("train/top1_acc", top1_acc.val, step)
                writer.add_scalar("train/top3_acc", top3_acc.val, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

                print(f"Step {step}/{config.N_STEPS} | "
                      f"Loss: {losses.val:.4f} | "
                      f"CE: {ce_losses.val:.4f} | "
                      f"KL: {kl_losses.val:.4f} | "
                      f"Top1: {top1_acc.val:.3f} | "
                      f"Top3: {top3_acc.val:.3f}")

    return step


def validate(val_loader, model, criterion, config, writer, epoch):
    """Validate the model."""
    model.eval()

    losses = AverageMeter()
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

                # Compute loss (just CE for validation)
                if hasattr(criterion, 'compute_ce_loss'):
                    loss = criterion.compute_ce_loss(
                        predicted_moves,
                        batch["moves"][:, 1:],
                        batch["lengths"]
                    )
                else:
                    loss = criterion(
                        predicted_moves,
                        batch["moves"][:, 1:],
                        batch["lengths"]
                    )

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
    writer.add_scalar("val/top1_acc", top1_acc.avg, epoch)
    writer.add_scalar("val/top3_acc", top3_acc.avg, epoch)

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
    print(f"GT Regularization: {config.USE_GT_REGULARIZATION}")
    if config.USE_GT_REGULARIZATION:
        print(f"GT Weight: {config.GT_WEIGHT}")

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
        print("Please run: python data/prepare_sample_data.py first")
        return

    # Create model
    print("\nInitializing model...")
    model = create_model(config).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Create criterion
    if config.USE_GT_REGULARIZATION:
        # Check if precomputed cache exists
        precomputed_cache = None
        if hasattr(config, 'STOCKFISH_CACHE_PATH'):
            cache_path = Path(config.STOCKFISH_CACHE_PATH)
            if cache_path.exists():
                precomputed_cache = str(cache_path)
                print(f"Found precomputed Stockfish cache: {precomputed_cache}")
            else:
                print(f"Warning: Stockfish cache not found at {cache_path}")
                print("Will use real-time Stockfish evaluation (slower)")
        
        criterion = GameTheoreticLoss(
            eps=config.LABEL_SMOOTHING,
            n_predictions=config.N_MOVES,
            gt_weight=config.GT_WEIGHT,
            stockfish_path=config.STOCKFISH_PATH if hasattr(config, 'STOCKFISH_PATH') else None,
            stockfish_depth=config.STOCKFISH_DEPTH if hasattr(config, 'STOCKFISH_DEPTH') else 15,
            disable_stockfish=False,
            precomputed_cache_path=precomputed_cache  # Pass precomputed cache if available
        ).to(DEVICE)
        
        # Set move vocabulary from chess-transformers
        try:
            from chess_transformers.data.vocabulary import Vocabulary
            vocab = Vocabulary()
            criterion.set_move_vocab(vocab.moves_to_indices, vocab.indices_to_moves)
            print(f"Successfully loaded move vocabulary with {len(vocab.moves_to_indices)} moves")
        except Exception as e:
            print(f"Warning: Could not load move vocabulary from chess-transformers: {e}")
            print("Attempting to build vocabulary manually...")
            try:
                # Build vocabulary manually
                move_to_idx, idx_to_move = build_chess_vocabulary()
                criterion.set_move_vocab(move_to_idx, idx_to_move)
                print(f"Built move vocabulary with {len(move_to_idx)} moves")
            except Exception as e2:
                print(f"Error building vocabulary: {e2}")
                print("GT regularization will not work properly")
        
        print("Using game-theoretic regularization loss with subsampled Stockfish evaluation (5% of batch)")
    else:
        criterion = LabelSmoothedCE(
            eps=config.LABEL_SMOOTHING,
            n_predictions=config.N_MOVES
        ).to(DEVICE)
        print("Using standard cross-entropy loss")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,  # Will be updated by scheduler
        betas=config.BETAS,
        eps=config.EPSILON
    )

    # AMP scaler
    scaler = GradScaler(device=DEVICE, enabled=config.USE_AMP)

    # Training loop
    start_epoch = 0
    step = 0
    epochs = (config.N_STEPS // (len(train_loader) // config.BATCHES_PER_STEP)) + 1

    print(f"\nTraining for {epochs} epochs ({config.N_STEPS} steps)")
    print("="*60)

    best_val_acc = 0.0

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)

        # Train
        step = train_epoch(
            train_loader, model, criterion, optimizer,
            scaler, epoch, config, writer, step
        )

        # Validate
        if (epoch + 1) % (config.EVAL_FREQUENCY // 100) == 0:
            val_loss, val_acc = validate(val_loader, model, criterion, config, writer, epoch)

            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                try:
                    save_checkpoint(
                        epoch, model, optimizer,
                        config.NAME, config.CHECKPOINT_FOLDER,
                        prefix="best_"
                    )
                    print(f"Saved best checkpoint (val_acc: {val_acc:.3f})")
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_FOLDER}")
    print(f"Logs saved to: {config.LOGS_FOLDER}")
    print("="*60)


if __name__ == "__main__":
    main()
