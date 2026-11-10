---
name: class-diversity
description: Calculate class diversity for categorical variables in a QUIQ-format table. Use for data quality assessment of categorical variable distributions, detecting overly homogeneous or heterogeneous value distributions, and generating per-variable diversity summaries.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files (class_diversity_total.txt, class_diversity_summary.csv, class_diversity_detail.txt).
    type: string
---

# Class Diversity

Calculates **class diversity** for each categorical variable in a QUIQ-format table. Used in the LYDUS quality management program to assess how evenly distributed the values of categorical variables are.

## When to Use This Skill

- After QUIQ conversion, to assess data quality of categorical variables
- To detect variables with poor value diversity (e.g., a single dominant class)
- To generate a per-variable diversity report for LYDUS quality review

## Input Requirements

A QUIQ-format DataFrame (or CSV) with the following columns used:
- `Mapping_info_1` — used to exclude `note`, `code`, `date` categories
- `Is_categorical` — must be `1` to be included
- `Value` — the categorical value
- `Original_table_name`, `Variable_name` — used to group variables

## Filtering Logic

Only variables meeting **all** of the following are analyzed:
1. `Mapping_info_1` does NOT contain `note`, `code`, or `date` (case-insensitive)
2. `Is_categorical == 1`
3. `Value` is not null

## Diversity Metrics

For each `(Original_table_name, Variable_name)` group:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `class_diversity` | `num_classes / total_count` | Ratio of unique values to total count |
| `shannon_diversity` | `-Σ p·log(p)` | Entropy-based diversity |
| `simpson_diversity_score` | `1 - Σ p²` (or `1` if class_diversity==1) | Probability that two random picks differ |
| `Class_diversity (%)` | `simpson_diversity_score × 100` | Final reported metric |

> **Simpson diversity score**: closer to 100% means more evenly distributed classes (higher diversity); closer to 0% means one class dominates.

## Output

| Output | Type | Description |
|--------|------|-------------|
| `results_df` | `pd.DataFrame` | One row per variable: table, variable, total count, class count, Class_diversity (%) |
| `dict_variable_counts` | `dict[str, pd.DataFrame]` | Per-variable class frequency table keyed by `"TABLE - variable"` |

When run as a script, also saves:
- `class_diversity_total.txt` — weighted average Class_diversity (%)
- `class_diversity_summary.csv` — `results_df` as CSV
- `class_diversity_detail.txt` — per-class counts for every variable

## How to Run

```python
import os
import duckdb

skill_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(skill_dir, "scripts/duckdb.sql")) as f:
    sql = f.read()

# {quiq_csv}: QUIQ CSV 파일 경로 (quiq 스킬 출력물)
quiq_csv = "/path/to/quiq_3patients.csv"
sql = sql.replace("{quiq_csv}", quiq_csv)

# 기본: 변수별 요약 (diversity_summary)
# 상세: SQL 마지막 SELECT를 class_detail 로 변경
df = duckdb.sql(sql).df()

# Weighted Class Diversity
total_num = df["Total Number of Data"].sum()
weighted = (df["Total Number of Data"] * df["Class_diversity (%)"]).sum() / total_num
print(f"Weighted Class Diversity (%) = {round(weighted, 2)}")

# Save
save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/class_diversity_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/class_diversity_total.txt", "w") as f:
    f.write(f"Weighted Class Diversity (%) = {round(weighted, 2)}\n")
print(f"Saved {len(df):,} rows → {save_path}")
```

## Critical Notes

1. **Simpson diversity score edge case** — when every value is unique (`class_diversity == 1`), score is set to `1` (100%) instead of the formula result, since each class has only one observation.

2. **Column name** — the output column is `'Class_diversity (%)'`. The visualization helper `draw_diversity_box_plot` expects this exact name.

3. **Performance** — iterates over all `(table, variable)` groups. For large QUIQ tables (millions of rows), this can be slow. Filter to specific tables beforehand if needed.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Simpson, E.H. (1949). Measurement of Diversity. *Nature*, 163, 688.
