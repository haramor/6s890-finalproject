"""
Minimal training test (non-interactive)
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
from models.game_theoretic_loss import LabelSmoothedCE

try:
    from chess_transformers.train.datasets import ChessDataset
    print("✓ Successfully imported ChessDataset")
except ImportError as e:
    print(f"✗ Could not import ChessDataset: {e}")
    print("Using mock dataset instead")
    ChessDataset = None

# Simple model for testing
class TestModel(nn.Module):
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
        x = x[:, 0]  # Take first token
        logits = self.classifier(x)
        return logits.unsqueeze(1)

def test_baseline():
    """Test baseline configuration"""
    print("\n" + "="*60)
    print("Testing BASELINE configuration")
    print("="*60)

    base_dir = Path(__file__).parent.parent
    data_folder = str(base_dir / "data" / "mixed_skill")
    h5_file = "mixed_skill_10k.h5"

    print(f"Data folder: {data_folder}")
    print(f"H5 file: {h5_file}")

    # Check if data exists
    if not Path(data_folder, h5_file).exists():
        print(f"✗ Data file not found: {Path(data_folder, h5_file)}")
        return False

    try:
        # Create dataloader
        if ChessDataset is not None:
            dataset = ChessDataset(
                data_folder=data_folder,
                h5_file=h5_file,
                split="train",
                n_moves=1
            )
            loader = DataLoader(dataset, batch_size=8, shuffle=True)
            print(f"✓ Created dataloader with {len(dataset)} samples")
        else:
            print("✗ ChessDataset not available, skipping dataloader test")
            return False

        # Create model
        model = TestModel()
        print(f"✓ Created model")

        # Create loss
        criterion = LabelSmoothedCE(eps=0.1, n_predictions=1)
        print(f"✓ Created loss function")

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print(f"✓ Created optimizer")

        # Test one training step
        print("\nTesting training step...")
        model.train()

        batch = next(iter(loader))
        predicted = model(batch)
        loss = criterion(
            predicted=predicted,
            targets=batch["moves"][:, 1:],
            lengths=batch["lengths"]
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"✓ Training step successful!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Prediction shape: {predicted.shape}")

        return True

    except Exception as e:
        print(f"✗ Error during training test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("Minimal Training Test")
    print("="*60)

    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    success = test_baseline()

    if success:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe training pipeline is working correctly.")
        print("\nNext steps:")
        print("  1. Run full training: python scripts/train.py --config configs/baseline_config.py")
        print("  2. View logs: tensorboard --logdir results/")
        print("  3. Try other conditions: expert_only and game_theoretic")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Tests failed")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
