#!/usr/bin/env python3
"""
Add val_split_index attribute to h5 files

This script adds the required val_split_index attribute to h5 files
that are missing it. The val_split_index determines where to split
the data between training and validation sets.
"""

import h5py
import sys

def add_val_split_index(h5_path, val_split_ratio=0.9):
    """
    Add val_split_index attribute to h5 file.
    
    Args:
        h5_path: Path to h5 file
        val_split_ratio: Fraction of data to use for training (default 0.9 = 90% train, 10% val)
    """
    print(f"Processing: {h5_path}")
    
    with h5py.File(h5_path, 'r+') as f:
        # Check if val_split_index already exists
        if 'val_split_index' in f['encoded_data'].attrs:
            print(f"  val_split_index already exists: {f['encoded_data'].attrs['val_split_index']}")
            return
        
        # Get the total number of games
        if 'moves' in f['encoded_data']:
            n_games = len(f['encoded_data']['moves'])
        elif 'board_positions' in f['encoded_data']:
            n_games = len(f['encoded_data']['board_positions'])
        else:
            print(f"  Error: Could not determine number of games")
            return
        
        # Calculate split index
        val_split_index = int(n_games * val_split_ratio)
        
        # Add attribute
        f['encoded_data'].attrs['val_split_index'] = val_split_index
        
        print(f"  Total games: {n_games}")
        print(f"  Train games: {val_split_index} ({val_split_ratio*100:.1f}%)")
        print(f"  Val games: {n_games - val_split_index} ({(1-val_split_ratio)*100:.1f}%)")
        print(f"  ✓ Added val_split_index = {val_split_index}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_h5_split.py <h5_file1> [h5_file2] ...")
        print("\nExample:")
        print("  python fix_h5_split.py data/rand_chunks_combined.h5 data/med_chunks_combined.h5")
        sys.exit(1)
    
    print("Adding val_split_index to h5 files...")
    print("=" * 60)
    
    for h5_path in sys.argv[1:]:
        try:
            add_val_split_index(h5_path)
            print()
        except Exception as e:
            print(f"  Error processing {h5_path}: {e}")
            print()
    
    print("=" * 60)
    print("Done!")
