# Project Roadmap

## ✅ Completed (Model Infrastructure)

- [x] Three experimental configurations
- [x] Game-theoretic loss function with Stockfish integration
- [x] Full training pipeline with TensorBoard logging
- [x] Sample data generation for testing
- [x] Training verified working (loss decreasing)

---

## 📋 For Progress Report (This Week)

- [ ] Run demo and document that infrastructure works
- [ ] Write up about:
  - Three experimental conditions
  - Game-theoretic loss formula
  - Training demo results (loss decreasing)
  - Next steps: scaling to real data

---

## 🎯 For Final Project

### 1. Data Acquisition 

**Download Lichess Database**
- Source: https://database.lichess.org/
- Filter tool: https://database.nikonoel.fr/
- Target: 50K-500K games per condition

**Create Two Datasets**
- **Mixed skill**: ELO 1500-2500, both players
- **Expert**: ELO 2500+, both players
- **Test set**: Hold out 10K games from 2500+ (no overlap with training)

**Convert to HDF5**
- Use chess-transformers data prep tools
- Format: Same as `prepare_sample_data.py` output
- Place in: `experiments/data/mixed_skill/` and `experiments/data/expert_2500/`

### 2. Full-Scale Training 

**Update Configs for Real Data**
```python
# In each config file:
H5_FILE = "real_data.h5"  # Change from synthetic
N_STEPS = 100000  # Scale up from 1000
BATCH_SIZE = 512  # Increase if GPU allows
```

**Run All Three Conditions**
```bash
python scripts/train.py --config configs/baseline_config.py
python scripts/train.py --config configs/expert_only_config.py
python scripts/train.py --config configs/game_theoretic_config.py
```

**Hyperparameter Tuning**
- GT regularization weight (λ): Try 0.01, 0.05, 0.1, 0.5, 1.0
- Stockfish depth: Test 10, 15, 20
- Learning rate schedule
- Label smoothing

**Multiple Seeds**
- Run each condition 3-5 times with different random seeds
- Compute mean and confidence intervals

### 3. Evaluation & Analysis 

**Sample Complexity Curves**
- Train on: 10K, 50K, 100K, 500K, 1M games
- Plot: Training samples (x) vs Top-1 accuracy (y)
- Compare all three conditions on same plot
- Measure: Games needed to reach 70% accuracy

**Stockfish Alignment**
- Compute KL-divergence on test set
- Compare model distributions vs Stockfish
- Report per condition

**Centipawn Loss**
- Integrate Stockfish evaluation in test loop
- Compute average CP loss per move
- Lower = better alignment with strong play

**Statistical Tests**
- Paired t-tests between conditions
- Confidence intervals on sample complexity
- Report significance levels

### 4. Visualization

**Key Plots**
- Sample complexity curves (all conditions)
- Training loss over time (CE + KL components)
- Top-1, Top-3, Top-5 accuracy curves
- Comparison tables with error bars

**Analysis**
- Where does GT regularization help most? (opening/middlegame/endgame)
- Example positions showing differences
- Error analysis

### 5. Writing

**Sections**
- Introduction & motivation
- Related work
- Methods (model, loss, training)
- Experiments & results
- Discussion & limitations
- Conclusion

---

## 🔧 Thoughts on optional Improvements

- [ ] Precompute Stockfish evaluations offline (faster training)
- [ ] Adaptive GT weight (anneal during training)
- [ ] Multi-task learning (predict outcome + moves)
- [ ] Attention visualization
- [ ] Compare other architectures (CNN, ViT)

---

## 📊 Key Metrics

1. **Sample Complexity**: Games needed for 70% top-1 accuracy
2. **Top-K Accuracy**: K=1, 3, 5
3. **Stockfish Alignment**: KL-divergence
4. **Centipawn Loss**: Average per move
5. **Statistical Significance**: p-values, confidence intervals


