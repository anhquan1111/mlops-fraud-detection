# Data Leakage: Diagnosis, Fix, and Measured Impact

**Status:** fixed · **Date:** 2026-08-21 · **Protocol tag in MLflow:** `leakfree_v2`

This document records two data-leakage defects that were present in the original
training pipeline, how they were found, what was changed, and — most importantly —
exactly how much the reported metrics moved once they were removed.

The headline is simple: **the numbers went down, and that is the correct result.**
A leak does not make a model better; it makes the *measurement* of the model
optimistic. Removing it does not cost performance, it reveals the performance that
was always there.

---

## 1. The two defects

### Leak A — the `Amount` scaler was fitted before the train/test split

**What the code did.** `src/features.py::preprocess()` fitted a `StandardScaler`
on the `Amount` column of the **entire** 284,807-row dataset, and `split_data()`
was called afterwards:

```python
# BEFORE — leaky
def preprocess(df):
    df = df.drop(columns=["Time"])
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])   # sees every row
    return df[FEATURE_COLS], df[TARGET_COL]

X, y = preprocess(df)
X_train, X_test, y_train, y_test = split_data(X, y)        # split comes second
```

**Why it is leakage.** `fit_transform` computes a mean and a standard deviation.
Both are statistics of the data, and both were computed over rows that later became
the test set. Every training row was therefore normalised using information derived
in part from the test set. At serving time no such statistic exists — production
data has not been seen yet — so the training distribution silently differs from the
serving distribution.

**Severity: low.** The leaked quantity is two scalars estimated from 284k rows, and
it touches 1 of 29 features. The full-data statistics (mean 88.3496, std 250.1197)
and the train-only statistics (mean 87.9702, std 245.5762) differ by roughly 0.4%
and 1.8% respectively. The effect on the reported metrics is small — but the defect
is unambiguous, and it is exactly the class of mistake this project's own
`AGENTS.md` forbids ("Áp dụng SMOTE trước khi split train/test — gây data leakage").
Fitting a scaler before the split is the same mistake wearing a different hat.

**A second, practical consequence.** Because the scaler was never persisted,
`src/api.py` re-implemented it with two hard-coded constants copied from an EDA
notebook. Nothing tied those constants to the model being served — a change in the
split or the seed would silently skew every production prediction with no test to
catch it.

### Leak B — early stopping selected the number of boosting rounds on the test set

**What the code did.** Both boosted-tree trainers passed the test set as the
early-stopping watch list:

```python
# BEFORE — leaky
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

**Why it is leakage.** Early stopping is model selection. It chooses one
hyperparameter — the number of trees — by scoring candidates and keeping the best.
Scoring them on the test set means the test set participated in choosing the model,
so the score it then reports is no longer an unbiased estimate of performance on
unseen data. The champion stopped at iteration 174 *because that iteration looked
best on the very rows used to report 0.8770*.

This is subtler than a duplicated row and correspondingly easier to ship. It is
also the more consequential of the two defects: it biases the metric that the
entire selection pipeline and the validation gate depend on.

**Severity: moderate.** Quantified in §4.

---

## 2. How they were found

Neither defect was found by a test, because no test asserted anything about
*ordering*. Both surfaced during a manual read of the pipeline in dependency order —
following the data from `load_data()` through to `model.fit()` and asking, at each
fitted transformation, *"what rows did this see?"*

Two questions did the work:

1. **"Which rows did this `.fit()` see?"** — Applied to the `StandardScaler` call,
   the answer was "all of them", and the split had not happened yet. That is Leak A.
2. **"Does anything choose between candidate models using test data?"** — Applied to
   `eval_set`, the answer was yes. That is Leak B.

The lesson worth carrying forward is that leakage is a property of *when* code runs
relative to the split, not of what any single line says. Reading the file top to
bottom is enough to find it; reading functions in isolation is not.

---

## 3. What changed

| Area | Before | After |
|---|---|---|
| Split | 80 / 20 (train / test) | **64 / 16 / 20** (train / val / test), stratified at each step |
| Split order | single split | **test carved out first**, val taken from the remainder |
| `Amount` scaling | fitted inside `preprocess()` on all rows | `preprocess()` returns raw `Amount`; `fit_amount_scaler()` fits on **train only**, `apply_amount_scaler()` transforms each split |
| Early stopping | `eval_set=[(X_test, y_test)]` | `eval_set=[(X_val, y_val)]` |
| Champion selection | ranked by test `pr_auc` | ranked by **`val_pr_auc`** |
| Validation gate input | test metrics | **validation metrics** (`val_*`) |
| Metrics logged | `pr_auc`, `recall`, … | **`val_*` and `test_*`** families, plus `tp`/`fp`/`fn`/`tn` counts |
| API scaling constants | hard-coded from full-data EDA | imported from `src/config.py`, asserted against the real train-split scaler by `tests/test_features.py` |
| Baseline reference | hard-coded `0.7156` in `train.py` | read from the LR run of the current experiment |

The **gate's comparison logic is unchanged**: a candidate is still required to be at
least as good as the model currently holding the `production` alias. Only the metric
family it reads has moved from test to validation.

One consequence of that had to be handled explicitly. Runs recorded before this fix
logged only a bare `pr_auc`, measured on a test set that had already driven early
stopping. That number is not on the same scale as `val_pr_auc`, so comparing them
would be meaningless in either direction. `_get_production_metrics()` therefore
detects a production run with no `val_pr_auc`, refuses the comparison, and reports
`FIRST_DEPLOYMENT` — resetting the baseline rather than fabricating a comparison.
Three tests in `TestProtocolChange` pin that behaviour, including one asserting that
the hard recall/precision floor still applies when the baseline is reset.

### The test set did not move

The new protocol carves out the test set with the same call, the same seed and the
same stratification as the old one:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

The 56,962 test rows (98 frauds) are therefore **the identical rows** used before the
fix — verified by comparing index sets. Every before/after test number below is
measured on exactly the same data, so the differences are attributable to the leak
and nothing else.

---

## 4. Measured impact

### 4a. The same model configuration, before and after

This is the clean measurement of what the leaks were worth: `lgbm_large`, identical
hyperparameters, identical test rows, trained under each protocol.

| Metric | Before (leakage) | After (leak-free) | Δ |
|---------------|-----------------|--------------|-------|
| PR-AUC | 0.8770 | 0.8703 | **−0.0067** |
| Recall | 0.8571 | 0.8469 | −0.0102 |
| Precision | 0.8485 | 0.7615 | **−0.0870** |
| F1 | 0.8528 | 0.8019 | −0.0509 |
| TP / FP | 84 / 15 | 83 / 26 | **+11 false alarms** |

Reading this honestly:

- **PR-AUC barely moved (−0.0067).** Ranking quality was genuinely there; the leaks
  were not manufacturing the model's ability to separate fraud from non-fraud.
- **Precision moved a lot (−0.087), and false positives went from 15 to 26.** This is
  where the optimism lived. Threshold-dependent metrics were flattered far more than
  the ranking metric, because early stopping was tuned against the exact rows those
  metrics were computed on.
- The practical lesson: **a leak's damage is not uniform across metrics.** Quoting
  only PR-AUC would have understated the problem by an order of magnitude.

### 4b. What the pipeline now ships

Selection also changed, because the gate now judges candidates on validation data.
On validation, `lgbm_large` misses the recall floor — 0.7975 against a required
0.80 — so it is no longer eligible. The gate's choice is `lgbm_regularized`.

| Metric | Old champion `lgbm_large` (leakage) | New champion `lgbm_regularized` (leak-free) | Δ |
|---------------|-----------------|--------------|-------|
| PR-AUC | 0.8770 | 0.7462 | −0.1308 |
| Recall | 0.8571 | **0.8878** | **+0.0307** |
| Precision | 0.8485 | 0.4555 | −0.3930 |
| F1 | 0.8528 | 0.6021 | −0.2507 |
| TP / FP | 84 / 15 | **87** / 104 | +3 caught, +89 false alarms |

All figures are on the held-out test set, scored once.

### 4c. Full run table (leak-free protocol)

Ranked by `val_pr_auc`, which is what selection uses. Test columns are reported, never selected on.

| Run | val PR-AUC | val Recall | val Prec | test PR-AUC | test Recall | test Prec | test F1 | TP | FP | FN | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lgbm_large` | 0.8160 | 0.7975 | 0.8289 | 0.8703 | 0.8469 | 0.7615 | 0.8019 | 83 | 26 | 15 | ❌ recall |
| `lgbm_default` | 0.8038 | 0.8228 | 0.4815 | 0.8496 | 0.8878 | 0.4652 | 0.6105 | 87 | 100 | 11 | ❌ precision |
| `xgb_default` | 0.7899 | 0.7848 | 0.8267 | 0.8604 | 0.8265 | 0.7714 | 0.7980 | 81 | 24 | 17 | ❌ recall |
| **`lgbm_regularized`** ⭐ | **0.7407** | **0.8354** | **0.5238** | 0.7462 | 0.8878 | 0.4555 | 0.6021 | 87 | 104 | 11 | ✅ **promoted** |
| `xgb_regularized` | 0.7135 | 0.7848 | 0.4593 | 0.7096 | 0.8571 | 0.4200 | 0.5638 | 84 | 116 | 14 | ❌ both |
| `xgb_deep` | 0.6831 | 0.7342 | 0.5000 | 0.6932 | 0.8061 | 0.4647 | 0.5896 | 79 | 91 | 19 | ❌ recall |
| `lr_baseline` | 0.6755 | 0.8861 | 0.0591 | 0.7105 | 0.9082 | 0.0606 | 0.1137 | 89 | 1379 | 9 | ❌ precision |

**The gate rejected 6 of 7 runs** — 4 on recall, 3 on precision (`xgb_regularized`
fails both). Under the leaky protocol it rejected 4 of 7.

Baseline for reference: Logistic Regression now scores **val PR-AUC 0.6755 /
test PR-AUC 0.7105**, against 0.7156 previously. The champion's improvement over
baseline is now **+0.0652 test PR-AUC (+9.2%)**, where the leaky pipeline reported
+22.6%. Most of that headline gain was measurement error.

---

## 5. An unresolved finding this fix exposed

The promoted champion clears the gate on validation (precision 0.5238) but **falls
below the same floor on the held-out test set (precision 0.4555)**.

This is not a bug in the fix — it is the fix working. A single 45,569-row validation
split containing 79 frauds is a noisy basis for a precision estimate, and selecting
the maximum over seven candidates on that split biases the winner upward. The test
set, untouched by selection, reports the shortfall.

Two things follow, and neither is a hyperparameter problem:

1. **The thresholds themselves were calibrated against leaked numbers.** `MIN_RECALL
   = 0.80` and `MIN_PRECISION = 0.50` were set when metrics were inflated. Under
   honest measurement the strongest models by test performance — `lgbm_large`
   (test precision 0.7615) and `xgb_default` (0.7714) — are rejected for missing
   recall by 0.003 and 0.015. The gate is now rejecting the models an operations
   team would most want. Re-deriving both thresholds from actual review capacity is
   a business conversation, and per `AGENTS.md` it is not a change to make
   unilaterally.
2. **Selection on a single validation split is too noisy at this fraud count.**
   Stratified k-fold cross-validation on train+val, selecting on the mean, would give
   a far more stable estimate than one 79-fraud split. That is the change most likely
   to close the val/test gap.

Deliberately **not** done: tuning hyperparameters until the numbers come back up.
That would re-introduce exactly the selection bias this document exists to remove —
the difference being that it would be invisible rather than logged.

### Threshold sweep, promoted champion, test set

`lgbm_regularized` is unusually threshold-sensitive, which is the real reason its
0.5 precision looks poor:

| Threshold | Recall | Precision | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.3 | 0.8878 | 0.2979 | 0.4462 | 87 | 205 | 11 |
| **0.5** (deployed) | **0.8878** | **0.4555** | **0.6021** | **87** | **104** | **11** |
| 0.6 | 0.8878 | 0.5472 | 0.6770 | 87 | 72 | 11 |
| 0.7 | 0.8878 | 0.6259 | 0.7342 | 87 | 52 | 11 |
| 0.8 | 0.8673 | 0.7025 | 0.7763 | 85 | 36 | 13 |
| 0.9 | 0.8265 | 0.7570 | 0.7902 | 81 | 26 | 17 |

Recall is completely flat from 0.3 to 0.7 while false positives fall from 205 to 52.
At threshold 0.7 this model would catch the same 87 frauds with 52 false alarms and
a test precision of 0.6259 — comfortably above the floor. The default of 0.5 is
simply not where this model should sit. Moving it remains a business decision
(`DECISION_THRESHOLD` in `src/config.py`), not one to make from the metrics alone.

---

## 6. Reproducing

```bash
uv run python src/train.py                        # 7 runs, logs val_* and test_*
uv run python scripts/select_best_model.py --dry-run   # gate decision, no promotion
uv run python scripts/select_best_model.py             # promote if the gate passes
uv run pytest tests/                                   # 73 tests
```

Split sizes: train 182,276 (315 frauds) · val 45,569 (79) · test 56,962 (98).
Amount scaler, fitted on train only: mean **87.9702**, std **245.5762** — mirrored in
`src/config.py` and asserted by `tests/test_features.py`.

---

## 7. What is now guarded by tests

Test count rose from 61 to **73**. The additions exist so these defects cannot
return silently:

- `preprocess()` must leave `Amount` raw.
- `fit_amount_scaler()` statistics must equal the training split's own mean/std, and
  must differ from statistics computed over all three splits.
- No index may appear in more than one split.
- Changing `val_size` must not move a single row into or out of the test set.
- `src/config.py` scaler constants must match the scaler fitted from the raw CSV.
- A production run without `val_pr_auc` must reset the baseline rather than be
  compared against, while the hard thresholds continue to apply.
