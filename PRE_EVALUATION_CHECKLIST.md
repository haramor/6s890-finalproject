# Pre-Evaluation Checklist

## ⚠️ CRITICAL ISSUES IN evaluate.py

Your `experiments/scripts/evaluate.py` has **placeholder methods that don't work**:

- ❌ `_reconstruct_fen()` (line 422) - always returns starting position
- ❌ `_move_to_index()` (line 428) - always returns None
- ❌ `_index_to_move()` (line 433) - always returns None

### Impact:
- ✅ **Top-K accuracy WILL WORK**
- ❌ **Stockfish alignment WILL FAIL**
- ❌ **Centipawn loss WILL FAIL**

---

## 🛡️ SAFE EVALUATION OPTIONS

### **Option 1: Use safe_evaluate.py (RECOMMENDED)**

I created `experiments/scripts/safe_evaluate.py` which:
- ✅ Only runs top-k accuracy (guaranteed to work)
- ✅ **Saves progress every 50 batches** (configurable)
- ✅ **Won't lose progress if it crashes**
- ✅ Processes batches one at a time (lower memory usage)

```bash
python experiments/scripts/safe_evaluate.py \
  --config experiments/configs/baseline_config.py \
  --checkpoint results/baseline_mixed_skill/checkpoints/best_checkpoint.pt \
  --output-dir results/evaluation_safe \
  --save-frequency 50
```

### **Option 2: Fix evaluate.py**

You need to implement these methods based on your data format:

1. **`_reconstruct_fen()`** - Convert batch data back to FEN string
2. **`_move_to_index()`** - Map chess.Move to vocabulary index
3. **`_index_to_move()`** - Map vocabulary index to chess.Move

(These require understanding your dataset format and vocabulary)

---

## ✅ PRE-EVALUATION CHECKLIST

Before running evaluation, verify:

### 1. **Checkpoint Exists**
```bash
ls -lh results/*/checkpoints/*.pt
```

Expected: You should see your trained checkpoint file

### 2. **Test Data Exists**
```bash
ls -lh experiments/data/mixed_skill/*.h5
```

Expected: You should see your HDF5 data file (e.g., `mixed_skill_10k.h5`)

### 3. **Config is Correct**
```bash
cat experiments/configs/baseline_config.py | grep -E "NAME|H5_FILE|DATA_FOLDER"
```

Expected:
- `NAME = "baseline_mixed_skill"` (or your experiment name)
- `H5_FILE` matches your actual file
- `DATA_FOLDER` points to correct directory

### 4. **GPU/CPU Setup**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected: Should print whether CUDA is available

### 5. **Test Quick Evaluation First**
```bash
# Run on just a few batches to test
python experiments/scripts/safe_evaluate.py \
  --config experiments/configs/baseline_config.py \
  --checkpoint results/baseline_mixed_skill/checkpoints/best_checkpoint.pt \
  --batch-size 32 \
  --save-frequency 5
```

Watch for errors. If this works, scale up to full evaluation.

---

## 📊 EXPECTED EVALUATION TIME

With your config settings:
- Batch size: 64
- Dataset: ~10K games
- Batches: ~156 batches

Expected time:
- **GPU**: 5-15 minutes
- **CPU**: 30-60 minutes

The evaluation will print progress and save intermediate results every 50 batches.

---

## 🚨 IF EVALUATION CRASHES

### Recovery Steps:

1. **Check for intermediate results:**
   ```bash
   ls -lh results/evaluation_safe/*.intermediate
   ```

2. **Check the error message** - Common issues:
   - Out of memory → Reduce `--batch-size`
   - Dataset not found → Check `DATA_FOLDER` and `H5_FILE` in config
   - Model load error → Checkpoint might be corrupted

3. **Resume from checkpoint** (if you added that feature):
   The intermediate file shows how far you got

---

## 📈 AFTER EVALUATION

### View Results:
```bash
cat results/evaluation_safe/baseline_mixed_skill_evaluation.json
```

### Generate Visualizations:
```bash
python experiments/scripts/visualizations.py \
  results/evaluation_safe \
  results/visualizations
```

---

## 💡 TIPS

1. **Start small**: Test with a small batch size first
2. **Monitor memory**: Use `nvidia-smi` (GPU) or `htop` (CPU)
3. **Check intermediate files**: Look for `.intermediate` files to see progress
4. **Don't panic**: If it crashes, you have intermediate results

---

## ⏱️ TIMELINE

For your situation (teammate training for 20 hours):

1. **Now**: Run `safe_evaluate.py` with `--save-frequency 10` on a small test
2. **If successful**: Run full evaluation with `--save-frequency 50`
3. **Total time**: ~15-30 minutes for full evaluation
4. **Generate plots**: ~1 minute

Good luck! 🍀
