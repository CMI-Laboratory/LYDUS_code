---
name: classification
description: Calculate classification metrics (Accuracy, Precision, Recall, F1, AUROC) per variable in a QUIQ-format table with Ground_truth labels. Use for data quality assessment when predicted values can be compared against ground truth, generating per-variable confusion matrices and ROC curves.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file containing both Value (predicted) and Ground_truth columns.
    type: string
  save_path:
    description: Directory path to save output files (classification_total.txt, classification_summary.csv, classification_plots/).
    type: string
---

# Classification Metrics

Calculates classification performance metrics for each variable in a QUIQ-format table by comparing `Value` (predicted) against `Ground_truth` (true label). Used in the LYDUS quality management program to evaluate how accurately recorded values match ground truth.

## When to Use This Skill

- QUIQ table has `Ground_truth` column filled with reference labels
- To evaluate per-variable classification quality (accuracy, AUC, etc.)
- To generate confusion matrices and ROC curves per variable

## Input Requirements

A QUIQ-format CSV with:
- `Value` — predicted/recorded value
- `Ground_truth` — true reference label (must have at least one non-null value)
- `Original_table_name`, `Variable_name` — used to group variables

Rows where either `Value` or `Ground_truth` is null are excluded.

## Metrics

For each `(Original_table_name, Variable_name)` group:

| Metric | Method | Note |
|--------|--------|------|
| `Accuracy` | `accuracy_score` | Overall correctness |
| `Precision` | `precision_score(average='macro')` | Per-class average |
| `Recall` | `recall_score(average='macro')` | Per-class average |
| `F1Score` | `f1_score(average='macro')` | Harmonic mean |
| `AUROC` | `roc_auc` via `LabelBinarizer` | Mean across classes (OvR) |

Final output also includes **weighted averages** of all metrics (weighted by number of samples per variable).

## Output

| File | Description |
|------|-------------|
| `classification_total.txt` | Weighted Accuracy / Precision / Recall / F1 / AUROC |
| `classification_summary.csv` | Per-variable metrics table |
| `classification_plots/{n}_{table}_{variable}.png` | Confusion matrix + ROC curve per variable |

## How to Run

```python
import os
import pandas as pd

skill_dir = os.path.dirname(os.path.abspath(__file__))

import sys; sys.path.insert(0, skill_dir)
from scripts.classification import get_classification, draw_auroc_variable_plot

quiq = pd.read_csv("/path/to/quiq.csv")
result_df, df_grouped, (w_acc, w_prec, w_rec, w_f1, w_auc) = get_classification(quiq)

save_path = "/path/to/output"
os.makedirs(f"{save_path}/classification_plots", exist_ok=True)

with open(f"{save_path}/classification_total.txt", "w") as f:
    f.write(f"Weighted Accuracy   = {w_acc}\n")
    f.write(f"Weighted Precision  = {w_prec}\n")
    f.write(f"Weighted Recall     = {w_rec}\n")
    f.write(f"Weighted F1score    = {w_f1}\n")
    f.write(f"Weighted AUROC      = {w_auc}\n")

result_df.to_csv(f"{save_path}/classification_summary.csv", index=False)

for n, (idx, target_df) in enumerate(df_grouped):
    y_true = target_df["Ground_truth"].astype(str)
    y_pred = target_df["Value"].astype(str)
    draw_auroc_variable_plot(save_path, n, idx[0], idx[1], y_true, y_pred, y_true.unique())

print(f"Saved → {save_path}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/classification.py --config config.yaml
```

## Critical Notes

1. **Ground_truth 필수** — `Ground_truth` 컬럼에 값이 없으면 AssertionError 발생.

2. **AUROC (binary)** — 이진 분류 시 `y_true_binary[:, 0]` 기준 단일 AUC 계산. 다중 클래스는 OvR(One-vs-Rest) 방식으로 클래스별 AUC 평균.

3. **플롯 파일명** — `re.sub(r'[\\/:*?"<>|]', ' ', name)`으로 특수문자 제거 후 저장.

4. **sklearn 의존성** — DuckDB SQL 변환 불가. scikit-learn, matplotlib 필요.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.
