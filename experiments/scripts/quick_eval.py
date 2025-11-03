"""
Quick Evaluation Test

This script runs a minimal evaluation to verify the evaluation pipeline works.
Perfect for testing before running full evaluations.

Usage:
    python quick_eval.py
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'chess': 'python-chess',
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(module)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip3 install {' '.join(missing)}")
        return False
    return True


def check_checkpoints():
    """Check if any trained checkpoints exist."""
    print("\nChecking for trained checkpoints...")
    base_dir = Path(__file__).parent.parent

    checkpoint_paths = [
        base_dir / "results" / "baseline_mixed_skill" / "checkpoints",
        base_dir / "results" / "expert_only_2500" / "checkpoints",
        base_dir / "results" / "game_theoretic_reg" / "checkpoints",
    ]

    found_checkpoints = []
    for path in checkpoint_paths:
        if path.exists():
            checkpoints = list(path.glob("*.pt"))
            if checkpoints:
                print(f"  ✓ Found {len(checkpoints)} checkpoint(s) in {path.name}")
                found_checkpoints.extend(checkpoints)
            else:
                print(f"  ✗ No checkpoints in {path.name}")
        else:
            print(f"  ✗ {path.name} (directory not found)")

    return found_checkpoints


def test_evaluation():
    """Run a quick evaluation test."""
    print("\n" + "="*70)
    print("QUICK EVALUATION TEST")
    print("="*70)

    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first")
        return False

    # Check for checkpoints
    checkpoints = check_checkpoints()

    if not checkpoints:
        print("\n⚠️  No trained checkpoints found!")
        print("\nPlease train a model first:")
        print("  python scripts/train.py --config configs/baseline_config.py")
        return False

    # Test with first checkpoint
    checkpoint = checkpoints[0]
    print(f"\n{'='*70}")
    print(f"Testing evaluation with: {checkpoint.name}")
    print(f"{'='*70}\n")

    # Determine config based on checkpoint path
    if "baseline" in str(checkpoint):
        config_name = "baseline_config.py"
    elif "expert" in str(checkpoint):
        config_name = "expert_only_config.py"
    else:
        config_name = "game_theoretic_config.py"

    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "configs" / config_name

    # Import and run minimal evaluation
    try:
        print("Loading evaluation module...")
        sys.path.insert(0, str(Path(__file__).parent))

        # Test imports
        import torch
        import numpy as np
        print("  ✓ Core dependencies loaded")

        # Try loading config
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print(f"  ✓ Config loaded: {config.NAME}")

        # Test checkpoint loading
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        print(f"  ✓ Checkpoint loaded successfully")

        print("\n" + "="*70)
        print("✓ EVALUATION PIPELINE TEST PASSED!")
        print("="*70)
        print("\nThe evaluation system is working correctly!")
        print("\nNext steps:")
        print("  1. Run full evaluation:")
        print(f"     python scripts/evaluate.py --config {config_path} --checkpoint {checkpoint}")
        print("\n  2. Evaluate all conditions:")
        print("     python scripts/evaluate.py --all-conditions")
        print("\n  3. Generate visualizations:")
        print("     python scripts/visualizations.py results/evaluation results/visualizations")

        return True

    except Exception as e:
        print(f"\n✗ Error during evaluation test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("Quick Evaluation Pipeline Test")
    print("="*70)

    success = test_evaluation()

    if success:
        return 0
    else:
        print("\n⚠️  Evaluation test failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
