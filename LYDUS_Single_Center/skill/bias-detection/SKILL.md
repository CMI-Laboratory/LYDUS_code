---
name: bias-detection
description: Detect demographic bias in LYDUS data quality metrics by computing all 16 QUIQ quality metrics (completeness, range/date/format/sequence validity, preciseness, fidelity, class/instance diversity, sentence/vocabulary diversity, logical accuracy, cross-sectional/time-series consistency, classification, note accuracy/fidelity) per Sex/Race/Age group and computing GDI (Group Disparity Index). Uses an on-prem local exo LLM (OpenAI-compatible; closed-network safe, no external API key).
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file.
    type: string
  via_path:
    description: Path to VIA (Variable Information Annotation) CSV file. Used by format-validity and cross-sectional-consistency.
    type: string
  save_path:
    description: Directory path to save output CSVs and sub-metric plots.
    type: string
  operation_type_manual:
    description: Passed to logical-accuracy skill. false = automatic mode. Default false.
    type: boolean
  automatic_num:
    description: Passed to logical-accuracy skill. Number of variables to analyze automatically. Default 5.
    type: integer
  recommend_num:
    description: Passed to logical-accuracy skill. Number of correlated variables to recommend. Default 5.
    type: integer
---

# Bias Detection

Detects **demographic bias** in data quality by running all 16 LYDUS quality metrics on patient subgroups (Sex, Race, Age) and comparing scores. The **GDI (Group Disparity Index)** quantifies how much any group deviates from the overall mean for each metric.

## When to Use This Skill

- After QUIQ conversion, to assess whether data quality differs across demographic groups
- To identify metrics where one group is systematically under-served or over-represented
- To generate an equity report for LYDUS quality management assessment
- Before clinical ML model training, to detect data quality biases that may propagate into model bias

## SQL Support

**Not applicable.** Orchestrates Python implementations of all 16 sub-metrics plus local exo LLM calls.

## Pipeline Overview

```
1. LLM identifies Sex / Race / BirthDate columns in QUIQ
2. Build per-patient demographics pivot (one row per Patient_id)
3. Compute Age_Group from BirthDate + Event_date
4. For each group in {Sex, Race, Age_Group}:
     Run all 16 quality metrics on the group-filtered QUIQ subset
5. Build results table: rows = metrics, cols = group labels + Mean + GDI
```

## 16 Sub-Metrics

| Metric | LLM | Skill |
|--------|:-------:|-------|
| Range Validity | ✗ | range-validity |
| Date Validity | optional | date-validity |
| Format Validity | optional | format-validity |
| Sequence Validity | exo LLM | sequence-validity |
| Completeness | ✗ | completeness |
| Logical Accuracy | exo LLM | logical-accuracy |
| Cross-Sectional Consistency | exo LLM | cross-sectional-consistency |
| Time Series Consistency | ✗ | time-series-consistency |
| Class Diversity | ✗ | class-diversity |
| Instance Diversity | ✗ | instance-diversity |
| Fidelity | ✗ | fidelity |
| Preciseness | ✗ | preciseness |
| Accuracy / Precision / Recall / F1 / AUROC | ✗ | classification |
| Note Fidelity | exo LLM | note-fidelity |
| Note Accuracy | exo LLM | note-accuracy |
| Vocabulary Diversity | ✗ | vocabulary-diversity |
| Sentence Diversity | ✗ | sentence-diversity |

## GDI (Group Disparity Index)

`GDI = max(|group_score − mean_score|)` across all groups for each metric.

- GDI = 0 → all groups have identical scores (no bias)
- High GDI → large disparity between groups for that metric

## Demographic Grouping

| Group | Column | Detection | Age Bins |
|-------|--------|-----------|----------|
| Sex | LLM-identified | biological sex variable | — |
| Race | LLM-identified | race variable | — |
| Age | BirthDate + Event_date | date of birth variable | 0-9, 10-19, …, 80+ |

Rows with null Sex/Race/Age_Group are dropped before group iteration.

## Output

| File | Description |
|------|-------------|
| `bias_sex_summary.csv` | Metrics × sex groups + Mean + GDI |
| `bias_race_summary.csv` | Metrics × race groups + Mean + GDI |
| `bias_age_summary.csv` | Metrics × age groups + Mean + GDI |
| Sub-metric plots | SHAP plots, AUROC plots, boxplots (saved to save_path) |

## How to Run

```python
import pandas as pd
from scripts.bias_detection import get_bias_detection

quiq = pd.read_csv("/path/to/quiq.csv")
via  = pd.read_csv("/path/to/via.csv")

config = {
    'save_path':            '/path/to/output',
    'operation_type_manual': False,
    'target_variable':      '',
    'automatic_num':        5,
    'recommend_num':        5,
}

df_sex, df_race, df_age = get_bias_detection(quiq, via, config)
print(df_sex)
```

### As a script with config

```yaml
# config.yaml
quiq_path:            /path/to/quiq.csv
via_path:             /path/to/via.csv
save_path:            /path/to/output
operation_type_manual: false
target_variable:      ""
automatic_num:        5
recommend_num:        5
```

```bash
python scripts/bias_detection.py --config config.yaml
```

## Critical Notes

1. **All 16 sub-metrics run per group** — each metric is wrapped in try/except; failures return `np.nan` without stopping the run. Expect long runtimes for large datasets or many groups.

2. **Skill dependency imports** — `bias_detection.py` dynamically adds sibling skills' `scripts/` directories to `sys.path` at runtime. All 17 sibling skills must be present under the same skills root (`~/.claude/skills/` or `src/m4/skills/lydus/`).

3. **원본 코드 버그 수정**:
   - `model_ver` / `api_key` 제거: `_llm_ask_column`이 로컬 exo LLM 호출로 대체됨
   - `get_time_series_consistency(config['save_path'], quiq)` 인자 순서 반전 → `(quiq, save_path)` 로 수정
   - `get_code_validity` 반환값 3개 언팩 → 2개(`validation_df, error_summary`)로 수정
   - `get_unstructured_fidelity` / `get_unstructured_accuracy` → M4 스킬의 `get_note_fidelity` / `get_note_accuracy` 로 교체
   - `config['save_path'] = -1` 하드코딩 → 실제 save_path 사용

4. **GDI 계산** — `df.sub(df['Mean'], axis=0).abs().max(axis=1)`: Mean 열을 먼저 계산 후 그룹 열들과의 최대 절댓값 편차. Mean 열 자체는 GDI 계산에서 제외.

5. **Age bins** — `pd.cut(..., right=False)`: 0-9 includes 0 and excludes 10. 80+ = [80, 120).

6. **Dependencies** — all 16 sub-metric skill dependencies (see each skill's SKILL.md). LLM 호출은 사내 로컬 exo(OpenAI 호환) — 외부 전송 없음, API 키 불필요.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Bias_Detection.py (이성민 작성)
