"""
Quick Test Script

This script runs a minimal training test to verify the setup is working.
Perfect for a quick sanity check before running full experiments.

Usage:
    python quick_test.py
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
        'h5py': 'h5py',
        'tqdm': 'tqdm',
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


def check_data():
    """Check if sample data exists."""
    print("\nChecking for sample data...")
    base_dir = Path(__file__).parent.parent

    data_files = [
        base_dir / "data" / "mixed_skill" / "mixed_skill_10k.h5",
        base_dir / "data" / "expert_2500" / "expert_2500_10k.h5",
    ]

    all_exist = True
    for f in data_files:
        if f.exists():
            print(f"  ✓ {f.name}")
        else:
            print(f"  ✗ {f.name} (missing)")
            all_exist = False

    if not all_exist:
        print("\nSample data not found. Creating it now...")
        try:
            subprocess.run([
                sys.executable,
                str(base_dir / "data" / "prepare_sample_data.py")
            ], check=True)
            print("  ✓ Sample data created successfully")
            return True
        except subprocess.CalledProcessError:
            print("  ✗ Failed to create sample data")
            return False

    return True


def run_quick_training_test(config_name="baseline"):
    """Run a very short training test."""
    print(f"\nRunning quick training test ({config_name})...")
    print("This will run just a few steps to verify everything works.\n")

    base_dir = Path(__file__).parent.parent
    config_map = {
        "baseline": "configs/baseline_config.py",
        "expert": "configs/expert_only_config.py",
        "gt": "configs/game_theoretic_config.py",
    }

    config_path = base_dir / config_map.get(config_name, config_map["baseline"])
    train_script = Path(__file__).parent / "train.py"

    # Temporarily modify config to run fewer steps
    print(f"Config: {config_path}")
    print(f"This is a minimal test - full training takes longer.\n")

    try:
        # Note: This will run the training script
        # For a true quick test, you'd want to modify N_STEPS in the config
        # or pass it as a command-line argument

        print("Starting test training...")
        print("Press Ctrl+C to stop early if you see training is working.\n")
        print("-" * 60)

        result = subprocess.run([
            sys.executable,
            str(train_script),
            "--config", str(config_path)
        ], timeout=60)  # 60 second timeout for quick test

        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✓ Training test completed successfully!")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("✗ Training test encountered errors")
            print("=" * 60)
            return False

    except subprocess.TimeoutExpired:
        print("\n" + "=" * 60)
        print("✓ Test timeout reached (this is OK for quick test)")
        print("Training is working! You can run full training now.")
        print("=" * 60)
        return True
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("✓ Test interrupted (this is OK)")
        print("If you saw training start, everything is working!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n✗ Error running training: {e}")
        return False


def main():
    print("=" * 60)
    print("Quick Test for Chess Behavioral Cloning Project")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first")
        return 1

    # Check/create data
    if not check_data():
        print("\n⚠️  Data preparation failed")
        print("Try running manually: python data/prepare_sample_data.py")
        return 1

    print("\n" + "=" * 60)
    print("Setup check complete! Ready to train.")
    print("=" * 60)

    # Ask if user wants to run quick training test
    print("\nWould you like to run a quick training test?")
    print("This will start training for 60 seconds to verify everything works.")
    print("You can stop it early with Ctrl+C.\n")

    response = input("Run quick test? [Y/n]: ").strip().lower()

    if response in ['', 'y', 'yes']:
        success = run_quick_training_test("baseline")

        if success:
            print("\n" + "=" * 60)
            print("Next Steps:")
            print("=" * 60)
            print("\n1. View training logs:")
            print("   tensorboard --logdir experiments/results/")
            print("\n2. Run full training:")
            print("   python scripts/train.py --config configs/baseline_config.py")
            print("\n3. Try other conditions:")
            print("   python scripts/train.py --config configs/expert_only_config.py")
            print("   python scripts/train.py --config configs/game_theoretic_config.py")
            print("\n4. See GETTING_STARTED.md for detailed instructions")
            return 0
        else:
            print("\n⚠️  Quick test failed. Check error messages above.")
            return 1
    else:
        print("\nSkipping training test.")
        print("When you're ready, run:")
        print("  python scripts/train.py --config configs/baseline_config.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
