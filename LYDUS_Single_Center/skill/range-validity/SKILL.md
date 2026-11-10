---
name: range-validity
description: Detect IQR outliers (1.5×IQR rule) in numeric variables of a QUIQ-format table and compute Range Validity (% of non-outlier values). SQL version available via DuckDB PERCENTILE_CONT. No API key required. Use for LYDUS data quality assessment of numeric value range plausibility.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
  weight:
    description: IQR multiplier for outlier bounds (default 1.5). Larger values = less sensitive.
    type: number
---

# Range Validity

Detects **IQR-based outliers** in numeric variables of a QUIQ-format table. A value is an outlier if it falls below `Q25 − weight×IQR` or above `Q75 + weight×IQR`. Range Validity (%) is the proportion of non-outlier values.

## When to Use This Skill

- After QUIQ conversion, to assess whether numeric measurements fall within plausible ranges
- To identify values that are statistically extreme relative to the variable's distribution
- As part of LYDUS quality management assessment

## SQL Support

**Available.** `scripts/duckdb.sql` uses `PERCENTILE_CONT` for IQR calculation — faster for large datasets. **Both** the SQL and Python paths produce the per-variable boxplot PNGs (the SQL path draws them with matplotlib from the pooled numeric values); the Python version (`scripts/range_validity.py`) additionally exports the full outlier CSV.

## Filtering Logic

| Condition | Value |
|-----------|-------|
| `Variable_type` | contains `numeric` (case-insensitive) |
| `Is_categorical` | = 0 |
| `Value` | numeric (non-null, parseable as float) |
| Minimum group size | > 1,000 rows per `Variable_name` |

> **Note**: IQR is computed per `Variable_name` pooled across all tables (same behavior as original). Outliers are then attributed to each `(Original_table_name, Variable_name)` pair.

## Outlier Detection

`IQR = Q75 − Q25`

| Bound | Formula |
|-------|---------|
| Lower | `Q25 − 1.5 × IQR` |
| Upper | `Q75 + 1.5 × IQR` |

`Range Validity (%) = (Total − Outlier_total) / Total × 100`

## Output

| File | Description |
|------|-------------|
| `range_validity_total.txt` | Overall Range Validity (%), Total Num, Outlier Num |
| `range_validity_summary.csv` | Per-(table, variable): Total_num, under/upper/total outlier counts and proportions, Range Validity (%) |
| `range_validity_outlier_total.csv` | All outlier rows with Direction ('under'/'upper') |
| `range_validity_boxplots/` | Per-variable boxplot PNGs (**always produced** — both SQL and Python paths) |

## How to Run

### SQL 버전 (빠름, boxplot 포함)

```python
import os
import re
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

skill_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(skill_dir, "scripts/duckdb.sql")) as f:
    sql = f.read()

quiq_csv = "/path/to/quiq_3patients.csv"
sql = sql.replace("{quiq_csv}", quiq_csv)

df = duckdb.sql(sql).df()

total_num = df["Total_num"].sum()
outlier_num = df["Outlier_total_num"].sum()
range_validity = round((total_num - outlier_num) / total_num * 100, 2)
print(f"Range Validity (%) = {range_validity}")

save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/range_validity_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/range_validity_total.txt", "w") as f:
    f.write(f"Range Validity (%) = {range_validity}\n")
    f.write(f"Total Num = {total_num}\n")
    f.write(f"Outlier Num = {outlier_num}\n")

# Boxplots — one PNG per (table, variable), same style as the Python version.
# Values are the pooled numeric values per Variable_name (matches IQR pooling).
box_dir = os.path.join(save_path, "range_validity_boxplots")
os.makedirs(box_dir, exist_ok=True)
flierprops = dict(marker="o", markerfacecolor="green", markersize=2,
                  linestyle="none", alpha=0.2)
for idx, row in df.reset_index(drop=True).iterrows():
    table_name, var_name = row["Original_table_name"], row["Variable_name"]
    vals = duckdb.sql(
        "SELECT TRY_CAST(Value AS DOUBLE) AS v "
        f"FROM read_csv_auto('{quiq_csv}', all_varchar=true) "
        f"WHERE Variable_name = '{var_name}' "
        "AND Variable_type ILIKE '%numeric%' "
        "AND TRY_CAST(Is_categorical AS INTEGER) = 0 "
        "AND TRY_CAST(Value AS DOUBLE) IS NOT NULL"
    ).df()["v"].tolist()

    fig = plt.figure(figsize=(3, 6))
    ax = fig.add_subplot(111)
    if vals:
        ax.boxplot(vals, flierprops=flierprops)
        ax.set_title(f"{table_name} - {var_name}", fontsize=10, fontweight="bold")
    else:
        ax.set_title(f"{table_name} - {var_name}\n(No data)", fontsize=14, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
    fig.set_tight_layout(True)
    safe_t = re.sub(r'[\\/:*?"<>|]', " ", str(table_name))
    safe_v = re.sub(r'[\\/:*?"<>|]', " ", str(var_name))
    fig.savefig(os.path.join(box_dir, f"{idx}_{safe_t}_{safe_v}.png"))
    plt.close(fig)

print(f"Saved {len(df):,} rows + {len(df)} boxplots → {save_path}")
```

### Python 버전 (boxplot + outlier 상세 포함)

```python
import pandas as pd
from scripts.range_validity import get_range_validity

quiq = pd.read_csv("/path/to/quiq.csv", low_memory=False)
outlier_df, summary_df, label_vs_boxplot = get_range_validity(quiq, weight=1.5)

total_num = summary_df['Total_num'].sum()
outlier_num = summary_df['Outlier_total_num'].sum()
range_validity = round((total_num - outlier_num) / total_num * 100, 2)
print(f"Range Validity (%) = {range_validity}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
weight:    1.5   # optional, default 1.5
```

```bash
python scripts/range_validity.py --config config.yaml
```

## Critical Notes

1. **IQR pooling** — IQR is computed on all rows for a `Variable_name` regardless of `Original_table_name`. This matches the original LYDUS implementation but means the same IQR bounds apply to all tables containing that variable.

2. **1,000-row threshold** — same group-size rule as preciseness: the filter applies per `Variable_name` total count (pooled). A variable with 600 rows in Table A and 500 rows in Table B (1,100 total) passes.

3. **weight 파라미터** — 기본값 1.5 (표준 Tukey fence). 민감도를 낮추려면 3.0 사용.

4. **원본 코드 개선 사항**:
   - `dict_dynamic` + `inplace` chained 수정 → `pool` 변수로 정리
   - bare `except:` → `except Exception`
   - `summary_df.loc[len(summary_df)] = list` → row dict 리스트로 수집 후 `pd.DataFrame`
   - `os.path.join` 사용 (문자열 연결 대신)
   - `weight` 파라미터 외부 노출 (원본 하드코딩 1.5)

5. **Dependencies** — `pandas`, `numpy`, `matplotlib` (Python); `duckdb` + `matplotlib` (SQL path — matplotlib is needed for the boxplots)

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Range_Validity.py (이성민 작성)
- Tukey, J.W. (1977). Exploratory Data Analysis.
