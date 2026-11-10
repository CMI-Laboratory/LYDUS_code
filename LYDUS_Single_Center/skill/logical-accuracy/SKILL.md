---
name: logical-accuracy
description: Detect logical outliers in clinical variables using a multi-model ensemble (Quantile Regression + Gradient Boosting + Autoencoder for numeric; OneClassSVM + IsolationForest + Autoencoder for categorical). Uses an on-prem local exo LLM (OpenAI-compatible; closed-network safe, no external API key). Use for LYDUS data quality assessment of logical consistency in event, diagnosis, prescription, and procedure variables.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files (logical_accuracy_summary.csv, logical_accuracy_total.txt, outlier_*.csv).
    type: string
  operation_type_manual:
    description: "True = manually specify target_variable; False = automatic top-N by count."
    type: boolean
  target_variable:
    description: Target variable name (used when operation_type_manual=True).
    type: string
  automatic_num:
    description: Number of top variables to analyze automatically (used when operation_type_manual=False). Default 5.
    type: integer
  recommend_num:
    description: Number of correlated variables recommended by LLM. Default 5.
    type: integer
---

# Logical Accuracy

Detects **logical outliers** in clinical variables using a multi-model anomaly detection ensemble. A value is flagged as an outlier only when **all three models** agree — minimizing false positives.

## When to Use This Skill

- After QUIQ conversion, to assess whether recorded clinical values are logically consistent with the patient's clinical context
- To detect physiologically implausible values (e.g., abnormal lab results inconsistent with other measurements)
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** This skill requires:
- local exo LLM calls for sex/birthdate variable identification and correlated-variable recommendation
- statsmodels Quantile Regression
- sklearn (GBR, OneClassSVM, IsolationForest, RobustScaler)
- PyTorch Autoencoder with early stopping

## Filtering Logic

| Category | `Mapping_info_1` | `Variable_type` | `Is_categorical` | Mode |
|----------|-----------------|----------------|-----------------|------|
| Event (numeric) | contains `event` | contains `numeric` | = 0 | `evaluate_mode=0` |
| Diagnosis | contains `diagnosis` | any | = 1 | `evaluate_mode=1` |
| Prescription (drug) | contains `prescription`, `Mapping_info_2` contains `drug` | any | = 1 | `evaluate_mode=1` |
| Procedure | contains `procedure` | any | = 1 | `evaluate_mode=1` |

## Two Operation Modes

| Parameter | Mode | Description |
|-----------|------|-------------|
| `operation_type_manual=True` | Manual | Analyze a single specified `target_variable` |
| `operation_type_manual=False` | Automatic | Analyze top-N variables by count |

## Pipeline

For each target variable:

1. **LLM** identifies sex variable and birthdate variable in the QUIQ data
2. **LLM** recommends `recommend_num` correlated variables
3. **Clinical context vector** is built by joining target + recommended variables on (Patient_id, Event_date ± 7 days)
4. **Outlier detection** based on `evaluate_mode`:

### evaluate_mode=0 (Numeric Event Variables)
- Quantile Regression (q=0.01, 0.99)
- Gradient Boosting Regressor (q=0.01, 0.99)
- Autoencoder (reconstruction error > 98th percentile)
- **Outlier**: outside ALL bounds simultaneously (upper OR lower)

### evaluate_mode=1 (Categorical Variables)
- One-Class SVM (nu=0.02, kernel=rbf)
- Isolation Forest (contamination=0.02)
- Autoencoder (mean reconstruction error > 98th percentile)
- **Outlier**: flagged by ALL three models simultaneously

## Autoencoder Architecture

```
Encoder: Linear(d → d//1.3) → Tanh → Linear(d//1.3 → d//2)
Decoder: Linear(d//2 → d//1.3) → Tanh → Linear(d//1.3 → d)
Optimizer: Adam (lr=0.001), Loss: MSE, Early stopping (patience=5, min_delta=0.001)
```

## Output

| File | Description |
|------|-------------|
| `logical_accuracy_total.txt` | Logical Accuracy (%), Total Num, Outlier Num |
| `logical_accuracy_summary.csv` | Per-variable: Total Num, Outlier Num, Logical Accuracy (%) |
| `outlier_{i}_{variable}.csv` | Outlier rows for each variable with outliers |

## How to Run

```python
import pandas as pd
from scripts.logical_accuracy import get_logical_accuracy

quiq = pd.read_csv("/path/to/quiq.csv")

var_list_target, dict_total, dict_outlier = get_logical_accuracy(
    quiq=quiq,
    operation_type_manual=False,
    target_variable="",        # ignored when operation_type_manual=False
    automatic_num=5,
    recommend_num=5
)
```

### As a script with config

```yaml
# config.yaml
quiq_path:            /path/to/quiq.csv
save_path:            /path/to/output
operation_type_manual: false
target_variable:      ""       # only needed when operation_type_manual=true
automatic_num:        5
recommend_num:        5
```

```bash
python scripts/logical_accuracy.py --config config.yaml
```

## Critical Notes

1. **All-models-agree criterion** — a value is an outlier only if all 3 models flag it. This is intentionally conservative to reduce false positives.

2. **Clinical context window** — correlated numeric variables are matched within ±7 days of the target measurement date (closest value used).

3. **Sex + birthdate enrichment** — LLM identifies these automatically. If not found, the context vector is built without them.

4. **Original code bug fixed** — line 661 in LYDUS_Logical_Accuracy.py had a missing comma: `'Outlier Num' 'Logical Accuracy (%)'` → Python string concatenation created column `'Outlier NumLogical Accuracy (%)'`. Fixed in this skill.

5. **Memory management** — `gc.collect()` is called after each major step. For large QUIQ tables (millions of rows), consider limiting `automatic_num`.

6. **Dependencies** — `statsmodels`, `scikit-learn`, `torch`, `numpy`, `pandas` (LLM: local exo LLM, OpenAI-compatible)

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Logical_Accuracy.py (이성민 작성)
