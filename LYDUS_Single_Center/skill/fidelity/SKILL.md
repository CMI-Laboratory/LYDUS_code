---
name: fidelity
description: Calculate structured fidelity for clinical variables in a QUIQ-format table. Measures how frequently each clinical concept (event, diagnosis, prescription, procedure) appears per patient. Use for data quality assessment of clinical record completeness and recording pattern consistency.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files (fidelity_total.txt, fidelity_summary.csv).
    type: string
---

# Fidelity

Calculates **structured fidelity** for clinical variables in a QUIQ-format table. Measures the average per-patient recording frequency for each clinical concept, grouped by category (`Mapping_info_1`).

## When to Use This Skill

- After QUIQ conversion, to assess how faithfully clinical events are recorded per patient
- To identify variables with unusually high or low recording frequency
- As part of LYDUS quality management assessment

## Categories and Filtering

| Category | `Mapping_info_1` | `Is_categorical` | Grouped by Value |
|----------|-----------------|-----------------|-----------------|
| Event | contains `event` | any | ❌ (전체 빈도만) |
| Diagnosis | contains `diagnosis` | = 1 | ✅ |
| Prescription | contains `prescription` | = 1 | ✅ |
| Procedure | contains `procedure` | = 1 | ✅ |

- Event는 값 구분 없이 환자별 발생 횟수만 집계
- 나머지 3개 카테고리는 `(Variable_name, Value)` 조합별로 집계

## Metrics

`(Original_table_name, Variable_name[, Value])` 그룹별:

| 컬럼 | 의미 |
|------|------|
| `Patient_num` | 해당 항목을 가진 환자 수 |
| `Mean` | 환자별 평균 기록 횟수 |
| `Std` | 환자별 기록 횟수의 표준편차 |

**Weighted Fidelity** = `Σ(Patient_num × Mean) / Σ(Patient_num)`

## Output

| File | Description |
|------|-------------|
| `fidelity_total.txt` | Weighted Fidelity (전체 가중 평균 빈도) |
| `fidelity_summary.csv` | 카테고리별 전체 결과 |

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

total_num = df["Patient_num"].sum()
weighted_fidelity = round(
    (df["Patient_num"] * df["Mean"]).sum() / total_num, 2
)
print(f"Weighted Fidelity = {weighted_fidelity}")

save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/fidelity_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/fidelity_total.txt", "w") as f:
    f.write(f"Weighted Fidelity = {weighted_fidelity}\n")
print(f"Saved {len(df):,} rows → {save_path}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/fidelity.py --config config.yaml
```

## Critical Notes

1. **Event Value = NULL** — Event 카테고리는 값 종류와 무관하게 발생 빈도만 집계하므로 `Value` 컬럼이 NULL.

2. **Is_categorical 타입** — 원본 코드에서 `df['Is_categorical'] == 1` 비교 전 타입 변환 없음. 스킬 버전에서 `pd.to_numeric(..., errors='coerce')` 및 SQL `TRY_CAST` 로 수정.

3. **Std = NULL** — 환자가 1명인 그룹은 표준편차가 NULL (샘플 표준편차 ddof=1).

4. **Weighted Fidelity 해석** — 값이 클수록 환자 1인당 평균적으로 더 많은 기록이 있음. 정상 범위는 데이터셋과 기관에 따라 다름.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Fidelity.py (이성민 작성)
