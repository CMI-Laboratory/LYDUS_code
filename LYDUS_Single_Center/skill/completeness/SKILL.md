---
name: completeness
description: Calculate data completeness (non-null rate) per variable in a QUIQ-format table. Use for data quality assessment to identify variables with missing values and compute overall completeness ratio.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files (completeness_total.txt, completeness_summary.csv).
    type: string
---

# Completeness

Calculates **data completeness** for each variable in a QUIQ-format table. Used in the LYDUS quality management program to measure how many records have non-null values for each variable.

## When to Use This Skill

- After QUIQ conversion, to assess how complete the data is per variable
- To identify variables with high null rates
- To compute an overall completeness ratio across all variables

## Input Requirements

A QUIQ-format CSV with:
- `Original_table_name`, `Variable_name` — used to group variables
- `Value` — NULL presence determines completeness

## Metrics

For each `(Original_table_name, Variable_name)` group:

| Metric | Formula |
|--------|---------|
| `Total_num` | Total row count |
| `Null_num` | Rows where `Value` is NULL |
| `Completeness (%)` | `(Total_num - Null_num) / Total_num × 100` |

Overall completeness = weighted average across all variables (by `Total_num`).

## Output

| File | Description |
|------|-------------|
| `completeness_total.txt` | Overall Completeness (%), Total_num, Null_num |
| `completeness_summary.csv` | Per-variable completeness table |

## How to Run

```python
import os
import duckdb

skill_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(skill_dir, "scripts/duckdb.sql")) as f:
    sql = f.read()

quiq_csv = "/path/to/quiq_3patients.csv"
sql = sql.replace("{quiq_csv}", quiq_csv)

df = duckdb.sql(sql).df()

total_num = df["Total_num"].sum()
null_num = df["Null_num"].sum()
completeness = round((total_num - null_num) / total_num * 100, 2)
print(f"Completeness (%) = {completeness}")

save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/completeness_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/completeness_total.txt", "w") as f:
    f.write(f"Completeness (%) = {completeness}\n")
    f.write(f"Total Num = {total_num}\n")
    f.write(f"Null Num = {null_num}\n")
print(f"Saved {len(df):,} rows → {save_path}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/completeness.py --config config.yaml
```

## Critical Notes

1. **Value 컬럼 기준** — completeness는 `Value` 컬럼의 NULL 여부만 판단. 빈 문자열(`""`)은 non-null로 계산됨.

2. **전체 completeness** — `completeness_total.txt`의 값은 변수별 가중 평균이 아니라, 전체 행 기준 단순 비율 (`(전체 - NULL) / 전체 × 100`).

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Completeness.py (이성민 작성)
