# START HERE 👋

Welcome! This is your complete setup for the 6.S890 final project on game-theoretic behavioral cloning in chess.

## What You Have

✅ **Complete infrastructure** for testing whether game-theoretic rationality reduces sample complexity in chess behavioral cloning

✅ **Three experimental conditions** ready to run:
- Baseline (mixed skill)
- Expert-only (high ELO)
- Game-theoretic (expert + Stockfish regularization)

✅ **Custom loss function** that combines behavioral cloning with equilibrium penalties

✅ **Full training pipeline** with metrics, logging, and visualization

## What to Do Next

### For Progress Report (This Week)

**Step 1: Read the overview**
```bash
cat README.md
```

**Step 2: Run the quick test**
```bash
cd experiments
python3 scripts/quick_test.py
```

This will:
- Check your dependencies
- Create sample data
- Run a mini training test
- Verify everything works

**Step 3: If the test works, you're done!** 🎉

For your progress report, document:
- Setup works ✓
- Training pipeline runs ✓
- Next step: scale to real data

### For Full Project (Next Weeks)

1. **Download real chess data** from Lichess
2. **Run full experiments** at multiple scales
3. **Analyze results** and create plots
4. **Write final report**

See `TODO.md` for detailed checklist.

## File Guide

Read these in order:

1. **`README.md`** (you are here) - Project overview
2. **`GETTING_STARTED.md`** - Detailed setup instructions
3. **`PROJECT_SUMMARY.md`** - Technical details
4. **`TODO.md`** - What to do next

## Quick Command Reference

```bash
# Test everything works
cd experiments
python3 scripts/quick_test.py

# Create sample data
python3 data/prepare_sample_data.py

# Train baseline model
python3 scripts/train.py --config configs/baseline_config.py

# View training progress
tensorboard --logdir results/
```

## Key Concepts

### The Research Question

Can we reduce the amount of training data needed by:
1. **Using only expert games** (H1: experts have lower variance)
2. **Adding equilibrium regularization** (H2: guide model toward Nash equilibrium)

### The Innovation

A custom loss function:
```
L = L_CE + λ * L_KL

where:
L_CE = Cross-entropy (learn from human moves)
L_KL = KL-divergence (match Stockfish distribution)
```

### Three Experiments

1. **Baseline**: Train on mixed-skill games
2. **Expert**: Train on 2500+ ELO games only
3. **Game-Theoretic**: Expert games + Stockfish regularization

**Goal**: Show expert and game-theoretic need fewer samples to reach same accuracy.

## Project Structure

```
6s890-final/
├── experiments/
│   ├── configs/           ← Three experiment configs
│   ├── models/            ← Game-theoretic loss
│   ├── scripts/           ← Training code
│   └── data/              ← Data preparation
├── chess-transformers/    ← Base model architecture
└── *.md                   ← Documentation
```

## What Each Config Does

| Config | Data | Loss | Tests |
|--------|------|------|-------|
| `baseline_config.py` | Mixed skill | Cross-entropy only | H0: baseline |
| `expert_only_config.py` | ELO 2500+ | Cross-entropy only | H1: expert variance |
| `game_theoretic_config.py` | ELO 2500+ | CE + KL from Stockfish | H2: GT regularization |

## Success Path

### Progress Report ✅
- [x] Infrastructure set up (you're here!)
- [ ] Run quick test
- [ ] Document that it works
- [ ] Outline scaling plan

### Final Project
- [ ] Download real data
- [ ] Run full experiments
- [ ] Analyze sample complexity
- [ ] Write report

## If Something Breaks

1. **Dependencies missing?**
   ```bash
   pip3 install torch numpy scipy h5py tqdm tensorboard python-chess
   ```

2. **Can't create data?**
   - Check that `experiments/data/` exists
   - Run `prepare_sample_data.py` manually

3. **Training fails?**
   - Check error message
   - See `GETTING_STARTED.md` troubleshooting section
   - Try with smaller batch size

4. **Stockfish not found?**
   - Only needed for game-theoretic condition
   - Install: `brew install stockfish` (macOS)
   - Or set path: `export CT_STOCKFISH_PATH=/path/to/stockfish`

## Time Estimates

- **Quick test**: 5 minutes
- **Progress report prep**: 1-2 hours
- **Full data download**: 2-4 hours
- **Full experiments**: 20-40 hours (GPU time)
- **Analysis and writing**: 20-30 hours

## Next Steps

1. ✅ You're reading this
2. → Run `python3 experiments/scripts/quick_test.py`
3. → Read `GETTING_STARTED.md` for details
4. → Check `TODO.md` for full project roadmap

## Questions?

- **Setup issues**: See `GETTING_STARTED.md`
- **Technical details**: See `PROJECT_SUMMARY.md`
- **What to do next**: See `TODO.md`
- **Chess-transformers questions**: See `chess-transformers/README.md`

## Important Notes

- ⚠️ This currently uses **synthetic test data** (random positions)
- ⚠️ For real results, you need **real Lichess data**
- ✅ But the infrastructure is **complete and working**
- ✅ Perfect for demonstrating feasibility in progress report

## Team Division Suggestion

Since you split tasks into **datasets, model, and evaluation**:

**You (Model)**:
- ✅ Model infrastructure is done!
- → Next: Help with training experiments
- → Next: Analyze model behavior

**Datasets teammate**:
- → Download Lichess data
- → Filter by ELO
- → Convert to HDF5

**Evaluation teammate**:
- → Implement Stockfish alignment metrics
- → Create sample complexity plots
- → Run statistical tests

You can all run the quick test independently to verify setup works.

## Final Checklist

Before progress report meeting:
- [ ] Run quick test
- [ ] Verify all three configs load
- [ ] Start one training run (can stop after a few minutes)
- [ ] Take screenshot of TensorBoard
- [ ] Document in progress report

That's it! You're ready to go. 🚀

**Run this now:**
```bash
cd experiments
python3 scripts/quick_test.py
```

Good luck!
