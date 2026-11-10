---
name: preciseness
description: Calculate numeric preciseness (decimal-place consistency) for numeric variables in a QUIQ-format table. Detects the actual decimal precision of each variable and measures last-digit distribution diversity using the Gini-Simpson index. No API key required. Use for LYDUS data quality assessment of numeric recording precision.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files (preciseness_total.txt, preciseness_summary.csv, preciseness_histograms/).
    type: string
---

# Preciseness

Measures the **decimal-place consistency** of numeric variables in a QUIQ-format table. For each numeric variable with >1,000 records, the skill detects the effective decimal precision and evaluates how evenly the last digit is distributed — a proxy for measurement precision quality.

## When to Use This Skill

- After QUIQ conversion, to assess whether numeric measurements are recorded at consistent decimal precision
- To detect variables where values are suspiciously rounded (e.g., all values end in 0 or 5)
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** The iterative decimal-scale detection loop (`Decimal` arithmetic, per-row modulo checks) cannot be easily expressed in SQL.

## Filtering Logic

| Condition | Value |
|-----------|-------|
| `Variable_type` | contains `numeric` (case-insensitive) |
| `Is_categorical` | = 0 |
| `Value` | not null |
| Minimum group size | > 1,000 rows per `Variable_name` |

## Algorithm

For each qualifying `(Original_table_name, Variable_name)` group:

### Step 1 — Round to 3 decimal places
All values are converted to `Decimal` and rounded to `0.001`.

### Step 2 — Detect effective decimal precision
Iterate multipliers exp = 3, 2, 1, 0, −1 (i.e., ×1000, ×100, ×10, ×1, ×0.1):
- If >99% of scaled values are divisible by 10 → the data is "coarser" than this scale, continue
- First scale where <99% are divisible → **that scale** is the effective decimal precision (`Decimal_num`)
- If all scales pass → `Decimal_num = −4` (data is integer-like or extremely coarse)

### Step 3 — Last-digit Gini-Simpson
At the detected scale, extract the last digit (0–9) of each value and compute the **Gini-Simpson diversity index**:

`Preciseness (%) = (1 − Σp²) × 100`

- High score (≈ 100%) → digits are evenly distributed → measurement values are precise and varied
- Low score (≈ 0%) → one digit dominates → values are heavily rounded (e.g., all end in 0)

### Weighted Preciseness
`Σ(Total_num × Preciseness (%)) / Σ(Total_num)` across all variables.

## Output

| File | Description |
|------|-------------|
| `preciseness_total.txt` | Weighted Preciseness (%) |
| `preciseness_summary.csv` | Per-variable: Total_num, Decimal_num, Preciseness (%) |
| `preciseness_histograms/` | Per-variable histogram PNG of last-digit distribution |

## How to Run

```python
import pandas as pd
from scripts.preciseness import get_preciseness

quiq = pd.read_csv("/path/to/quiq.csv")
summary_df, histogram_values = get_preciseness(quiq)

total_num = summary_df['Total_num'].sum()
weighted = round((summary_df['Total_num'] * summary_df['Preciseness (%)']).sum() / total_num, 2)
print(f"Preciseness (%) = {weighted}")
print(summary_df)
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/preciseness.py --config config.yaml
```

## Critical Notes

1. **Minimum 1,000 rows required** — variables with ≤1,000 rows are excluded. This threshold ensures the last-digit distribution is statistically meaningful.

2. **Decimal_num interpretation**:
   - `3` → values recorded to 3 decimal places (e.g., 1.234)
   - `0` → integer-level precision (e.g., 42)
   - `−1` → rounded to tens (e.g., 40, 50, 60)
   - `−4` → sentinel for extremely coarse data (all scales >99% divisible)

3. **원본 코드 개선 사항**:
   - `totaltable['col'][i] = val` chained indexing (SettingWithCopyWarning) → `summary_df` 를 row dict 리스트로 구성 후 `pd.DataFrame(rows)` 로 변경
   - `draw_histogram` 내 `os.path.join` 사용 (문자열 연결 대신)
   - 내부 for 루프의 `dftemp2` 재사용 패턴 → `_detect_decimal_scale` 함수로 분리

4. **Dependencies** — `pandas`, `numpy`, `matplotlib`, `decimal` (stdlib)

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Preciseness.py (이성민 작성)
- Simpson, E.H. (1949). Measurement of Diversity. *Nature*, 163, 688.
