#!/bin/bash
# Quick activation script for the project

echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated!"
echo ""
echo "Quick commands:"
echo "  cd experiments && python scripts/demo_training.py    # Run 5-step demo"
echo "  cd experiments && python scripts/train.py --config configs/baseline_config.py  # Full training"
echo "  tensorboard --logdir experiments/results/             # View logs"
echo ""
echo "Current Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Ready to go! 🚀"
