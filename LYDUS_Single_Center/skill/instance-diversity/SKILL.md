---
name: instance-diversity
description: Calculate instance diversity and Gini-Simpson index per (variable, value) combination in a QUIQ-format table. Measures how diversely distributed patients are across occurrences of each clinical value. Use for data quality assessment of recording pattern diversity in event, diagnosis, prescription, and procedure variables.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files (instance_diversity_total.txt, instance_diversity_summary.csv).
    type: string
---

# Instance Diversity

Calculates **instance diversity** and **Gini-Simpson index** per `(Original_table_name, Variable_name, Value)` combination in a QUIQ-format table. Measures how evenly distributed patients are in terms of how many times they contribute to each clinical value.

## When to Use This Skill

- After QUIQ conversion, to assess whether specific clinical values are dominated by a few patients (low diversity) or spread across many patients (high diversity)
- To detect recording bias — e.g., one patient contributing disproportionately many records for a given value
- As part of LYDUS quality management assessment

## Filtering Logic

| Category | `Mapping_info_1` | `Mapping_info_2` | `Is_categorical` |
|----------|-----------------|-----------------|-----------------|
| Event | contains `event` | any | = 1 |
| Diagnosis | contains `diagnosis` | any | = 1 |
| Prescription | contains `prescription` | contains `drug` | = 1 |
| Procedure | contains `procedure` | any | = 1 |

## Metrics

For each `(Original_table_name, Variable_name, Value)` group:

1. **Per-patient occurrence count**: group by `(ClassKey, Patient_id)` → `Instance_Count` per patient
2. **Instance Diversity (%)**: `num_patients / total_occurrences × 100`
   - 높을수록 환자당 평균 발생 횟수가 적음 (여러 환자에 분산)
3. **Gini-Simpson Index (%)**: `(1 - Σp²) × 100`
   - `p = patients_with_count_k / total_occurrences`
   - 환자별 발생 횟수 분포의 Simpson 다양성
   - `instance_diversity == 1` (모든 환자 1회 발생) → score = 100%

**Weighted averages**: weighted by total occurrence count across all ClassKey groups.

## Output

| File | Description |
|------|-------------|
| `instance_diversity_total.txt` | Weighted Instance Diversity (%), Weighted Gini-Simpson Index (%) |
| `instance_diversity_summary.csv` | Per-(table, variable, value): num_patients, total_num, Instance Diversity (%), Gini-Simpson Index (%) |

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

total_items = df["total_num"].sum()
weighted_instance = round((df["total_num"] * df["Instance Diversity (%)"]).sum() / total_items, 2)
weighted_simpson  = round((df["total_num"] * df["Gini-Simpson Index (%)"]).sum() / total_items, 2)
print(f"Weighted Instance Diversity (%) = {weighted_instance}")
print(f"Weighted Gini-Simpson Index (%) = {weighted_simpson}")

save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/instance_diversity_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/instance_diversity_total.txt", "w") as f:
    f.write(f"Weighted Instance Diversity (%) = {weighted_instance}\n")
    f.write(f"Weighted Gini-Simpson Index (%) = {weighted_simpson}\n")
print(f"Saved {len(df):,} rows → {save_path}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/instance_diversity.py --config config.yaml
```

## Critical Notes

1. **Class Diversity vs Instance Diversity** — class diversity는 한 변수 내 값들의 분포를 측정; instance diversity는 특정 값에 기여하는 환자들의 발생 빈도 분포를 측정.

2. **Prescription 필터** — `Mapping_info_2 LIKE '%drug%'` 조건이 추가됨 (처방 정보 중 약물명만 대상).

3. **원본 오타 수정** — `'Weighted Instance Diveristy (%)'` → `'Weighted Instance Diversity (%)'`

4. **None 처리** — 환자가 없거나 총 발생 수가 0인 ClassKey는 `None`으로 반환됨. 가중 평균 계산 시 0으로 처리.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Instance_Diversity.py (이성민 작성)
- Simpson, E.H. (1949). Measurement of Diversity. *Nature*, 163, 688.
