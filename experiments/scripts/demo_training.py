"""
Demo training script - runs just a few steps to demonstrate pipeline works
"""
import sys
import os
from pathlib import Path

# Add paths
chess_transformers_path = Path(__file__).parent.parent.parent / "chess-transformers"
sys.path.insert(0, str(chess_transformers_path))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import importlib.util

from models.game_theoretic_loss import LabelSmoothedCE
from chess_transformers.train.datasets import ChessDataset

# Load config
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 128)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=2
        )
        self.classifier = nn.Linear(128, 1968)

    def forward(self, batch):
        board = batch["board_positions"]
        x = self.embedding(board)
        x = self.transformer(x)
        x = x[:, 0]
        logits = self.classifier(x)
        return logits.unsqueeze(1)

def main():
    print("="*70)
    print("DEMO TRAINING - Baseline Configuration")
    print("="*70)

    # Load config
    base_dir = Path(__file__).parent.parent
    config = load_config(base_dir / "configs" / "baseline_config.py")

    print(f"\nConfiguration:")
    print(f"  Experiment: {config.NAME}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Data: {config.H5_FILE}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create dataloader
    print(f"\nLoading data...")
    train_loader = DataLoader(
        ChessDataset(
            data_folder=config.DATA_FOLDER,
            h5_file=config.H5_FILE,
            split="train",
            n_moves=config.N_MOVES
        ),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    print(f"  ✓ Train samples: {len(train_loader.dataset)}")

    # Create model
    print(f"\nInitializing model...")
    model = SimpleModel()
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Create loss
    criterion = LabelSmoothedCE(eps=0.1, n_predictions=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(device='cpu', enabled=False)

    print(f"\n{'='*70}")
    print(f"Starting training demonstration (5 steps)...")
    print(f"{'='*70}\n")

    model.train()
    step = 0
    max_steps = 5

    for i, batch in enumerate(train_loader):
        if step >= max_steps:
            break

        # Forward pass
        predicted = model(batch)
        loss = criterion(
            predicted=predicted,
            targets=batch["moves"][:, 1:],
            lengths=batch["lengths"]
        )

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Batch: {batch['board_positions'].shape[0]}")

    print(f"\n{'='*70}")
    print(f"✓ TRAINING DEMO COMPLETE!")
    print(f"{'='*70}")
    print(f"\nThe pipeline is working successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run full training: python scripts/train.py --config configs/baseline_config.py")
    print(f"  2. Try expert-only: python scripts/train.py --config configs/expert_only_config.py")
    print(f"  3. Try game-theoretic: python scripts/train.py --config configs/game_theoretic_config.py")
    print(f"  4. View logs: tensorboard --logdir results/")

if __name__ == "__main__":
    main()
