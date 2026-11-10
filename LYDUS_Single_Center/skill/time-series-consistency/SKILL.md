---
name: time-series-consistency
description: Detect temporal distribution shifts (change points) in clinical data across years using GradientBoosting classifier + SHAP. For each year, trains a model to distinguish year-1 vs year data — AUROC ≥ 0.8 indicates a change point. No API key required. Use for LYDUS data quality assessment of temporal consistency across data collection periods.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file. Must contain data spanning multiple years with non-null Event_date.
    type: string
  save_path:
    description: Directory path to save output files and plots.
    type: string
---

# Time Series Consistency

Detects **temporal distribution shifts** (change points) in clinical data by training a GradientBoosting classifier to distinguish consecutive year pairs. If AUROC ≥ 0.8, the distribution between year-1 and year differs significantly — indicating a change point (e.g., coding practice change, dataset collection change).

## When to Use This Skill

- After QUIQ conversion, to detect whether clinical variable distributions have shifted over time
- To identify years with sudden changes in documentation or coding patterns
- To assess longitudinal stability of data collection
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** Requires sklearn GradientBoostingClassifier and SHAP.

## Categories and Pivot Logic

| Category | Filter | Pivot Column |
|----------|--------|-------------|
| Event | `Mapping_info_1` contains `event` AND `Mapping_info_2` contains `lab_event` | `Variable_name` (binary presence per patient-day) |
| Diagnosis | `Mapping_info_1` contains `diagnosis` | `Value` (binary presence per patient-day) |
| Prescription | `Mapping_info_1` contains `prescription` AND `Mapping_info_2` contains `drug` | `Value` |
| Procedure | `Mapping_info_1` contains `procedure` | `Value` |

## Change Point Detection Algorithm

For each year in `[min_year+1, max_year-1]`:

1. Select records from `year-1` (label=0) and `year` (label=1)
2. Skip if: no data, class imbalance (`min_class / total < 0.25`), or too few samples
3. Train `GradientBoostingClassifier` (50/50 train/test split, stratified)
4. Compute AUROC on test set
5. **AUROC ≥ 0.8** → change point; save SHAP bar plot

`Time Series Consistency (%) = (total_years − change_points) / total_years × 100`

## Output

| File | Description |
|------|-------------|
| `time_series_consistency_total.txt` | Overall Consistency (%), Total Time Points, Change Points |
| `time_series_consistency_summary.csv` | Per-category: Total_time_point, Change_point, Time Series Consistency (%) |
| `{Category} - AUROC plot.png` | Year vs AUROC line plot with change points in red |
| `{Category} ({Year}) - SHAP Plot.png` | SHAP bar plot for each detected change point |

## How to Run

```python
import pandas as pd
from scripts.time_series_consistency import get_time_series_consistency

quiq = pd.read_csv("/path/to/quiq.csv")
save_path = "/path/to/output"

total_results, summary = get_time_series_consistency(quiq, save_path)

valid = summary.dropna(subset=['Total_time_point'])
total_tp = valid['Total_time_point'].sum()
change_pt = valid['Change_point'].sum()
consistency = round((total_tp - change_pt) / total_tp * 100, 2)
print(f"Time Series Consistency (%) = {consistency}")
print(summary)
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/time_series_consistency.py --config config.yaml
```

## Critical Notes

1. **Multi-year data required** — needs at least 3 distinct years per category (min_year+1 to max_year-1). Single-year or two-year data returns no results.

2. **원본 코드 버그 수정 (Prescription 필터)**:
   - 원본: `Mapping_info_1.str.contains('drug')` → `Mapping_info_1='prescription'`이라 항상 빈 결과
   - 수정: `Mapping_info_2.str.contains('drug')` → 처방 약물명 행만 올바르게 필터

3. **Change point ≠ data error** — AUROC ≥ 0.8은 연도간 분포 차이가 크다는 의미. 실제 임상 변화(신규 약물, ICD 개정 등)가 원인일 수 있으므로 맥락 해석 필요.

4. **SettingWithCopyWarning 수정** — 필터된 슬라이스에 직접 컬럼 추가하던 원본 코드 → 모두 `.copy()` 후 처리.

5. **리팩터링** — 4개 카테고리의 거의 동일한 블록 → `_run_category` + `_build_pivot_and_years` + `_detect_change_points` 함수로 통합.

6. **Dependencies** — `scikit-learn`, `shap`, `pandas`, `numpy`, `matplotlib`

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Time_Series_Consistency.py (이성민 작성)
- Lundberg, S.M., Lee, S-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
