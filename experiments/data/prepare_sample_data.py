"""
Sample Data Preparation Script

This script creates a small synthetic dataset for testing the training pipeline.
For real experiments, you'll need to:
1. Download PGN files from Lichess database
2. Filter by ELO rating using pgn-extract
3. Convert to HDF5 format using chess-transformers data prep tools

For now, this creates a minimal dataset to verify the pipeline works.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys

def create_sample_h5_file(
    output_path: str,
    num_samples: int = 1000,
    train_split: float = 0.9
):
    """
    Create a minimal HDF5 file with random chess data for testing.

    In production, you'll use real chess data from Lichess/TWIC.

    Args:
        output_path: Where to save the H5 file
        num_samples: Number of sample positions to generate
        train_split: Fraction of data for training (rest for validation)
    """
    print(f"Creating sample dataset with {num_samples} positions...")

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Create dataset
        dt = np.dtype([
            ('board_position', 'i4', (64,)),  # 64 squares
            ('turn', 'i4'),
            ('white_kingside_castling_rights', 'i4'),
            ('white_queenside_castling_rights', 'i4'),
            ('black_kingside_castling_rights', 'i4'),
            ('black_queenside_castling_rights', 'i4'),
            ('moves', 'i4', (11,)),  # <move> + 10 future moves
            ('length', 'i4'),
        ])

        dset = f.create_dataset('encoded_data', (num_samples,), dtype=dt)

        # Generate random data
        for i in range(num_samples):
            # Random board position (0-12: empty, pieces)
            board = np.random.randint(0, 13, size=64, dtype=np.int32)

            # Random turn (0: white, 1: black)
            turn = np.random.randint(0, 2, dtype=np.int32)

            # Random castling rights (0: no, 1: yes)
            castling = np.random.randint(0, 2, size=4, dtype=np.int32)

            # Random moves (1-1968: UCI move indices, 0 is <move> token)
            moves = np.zeros(11, dtype=np.int32)
            moves[0] = 0  # <move> token
            moves[1:] = np.random.randint(1, 1968, size=10)

            # Random sequence length (1-10)
            length = np.random.randint(1, 11, dtype=np.int32)

            # Store
            dset[i] = (
                board, turn,
                castling[0], castling[1], castling[2], castling[3],
                moves, length
            )

        # Add metadata
        dset.attrs['val_split_index'] = int(num_samples * train_split)

        print(f"Created dataset with:")
        print(f"  - Total samples: {num_samples}")
        print(f"  - Train samples: {dset.attrs['val_split_index']}")
        print(f"  - Val samples: {num_samples - dset.attrs['val_split_index']}")
        print(f"  - Saved to: {output_path}")


def create_sample_datasets():
    """Create sample datasets for all three experimental conditions."""

    base_dir = Path(__file__).parent.parent

    # Create directories
    (base_dir / "data" / "mixed_skill").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "expert_2500").mkdir(parents=True, exist_ok=True)

    # Create mixed skill dataset (1000 samples for quick testing)
    print("\n" + "="*60)
    print("Creating BASELINE (mixed skill) dataset")
    print("="*60)
    create_sample_h5_file(
        output_path=str(base_dir / "data" / "mixed_skill" / "mixed_skill_10k.h5"),
        num_samples=1000,
        train_split=0.9
    )

    # Create expert dataset (1000 samples for quick testing)
    print("\n" + "="*60)
    print("Creating EXPERT-ONLY dataset")
    print("="*60)
    create_sample_h5_file(
        output_path=str(base_dir / "data" / "expert_2500" / "expert_2500_10k.h5"),
        num_samples=1000,
        train_split=0.9
    )

    print("\n" + "="*60)
    print("Sample data preparation complete!")
    print("="*60)
    print("\nNOTE: This is SYNTHETIC data for testing the pipeline.")
    print("For real experiments, you need to:")
    print("  1. Download Lichess database from https://database.lichess.org/")
    print("  2. Filter by ELO using pgn-extract or database.nikonoel.fr")
    print("  3. Convert to HDF5 using chess-transformers data prep")
    print("\nSee README.md for details.")


if __name__ == "__main__":
    create_sample_datasets()
