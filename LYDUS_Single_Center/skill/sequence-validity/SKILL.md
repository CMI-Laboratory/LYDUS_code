---
name: sequence-validity
description: Validate temporal ordering of date variables in a QUIQ-format table. Uses a local exo LLM to automatically identify start/end date variable pairs, then checks whether start_date <= end_date for each matched record. Runs on an on-prem local exo LLM (closed-network safe; no external API key). Use for LYDUS data quality assessment of chronological consistency.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file. Must contain rows where Mapping_info_1 contains 'date'.
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
---

# Sequence Validity

Validates the **chronological ordering** of date variables in a QUIQ-format table. A local exo LLM automatically identifies meaningful (start, end) date pairs (e.g., admission → discharge), then checks whether `start_date ≤ end_date` for each matched patient record.

## When to Use This Skill

- After QUIQ conversion, to detect records where events occur in impossible or illogical order (e.g., discharge before admission)
- To assess chronological consistency of temporal data
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** The local exo LLM is required to identify date-variable pairs.

## Filtering Logic

| Condition | Value |
|-----------|-------|
| `Mapping_info_1` | contains `date` (case-insensitive) |
| `Value` | parsed as datetime |

## Pipeline

1. **Extract date rows** — filter `Mapping_info_1` contains `date`, parse `Value` as datetime
2. **Collect unique identifiers** — `Original_table_name - Variable_name` for all date variables
3. **LLM pair identification** — send identifier list to the local exo LLM; receive `timepoint_pairs` list
4. **LLM exclusion rules**:
   - Non-time variables excluded
   - Unpaired time variables excluded
   - Sensitive/complex variables excluded (death_time, year_of_birth, diagnosis_date, etc.)
   - Additional-context-required pairs excluded
5. **Validation** — for each pair: merge on `(Patient_id, Original_table_name, Primary_key)`, check `Start_date ≤ End_date`
6. **Summary** — per-(table, start_var, end_var): Total_num, Invalid_num, Sequence_Validity (%)

## Output

| File | Description |
|------|-------------|
| `sequence_validity_total.txt` | Overall Sequence Validity (%), Total Num, Invalid Num |
| `sequence_validity_summary.csv` | Per-(table, start_var, end_var): counts and Sequence_Validity (%) |
| `sequence_validity_detail.csv` | Per-record: Start_date, End_date, Is_valid |

## How to Run

```python
import pandas as pd
from scripts.sequence_validity import get_sequence_validity

quiq = pd.read_csv("/path/to/quiq.csv")

df_total, df_summary = get_sequence_validity(
    quiq=quiq
)

total_num = df_summary['Total_num'].sum()
invalid_num = df_summary['Invalid_num'].sum()
seq_validity = round((total_num - invalid_num) / total_num * 100, 2)
print(f"Sequence Validity (%) = {seq_validity}")
print(df_summary)
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/sequence_validity.py --config config.yaml
```

## Critical Notes

1. **Same-table constraint** — pairs spanning different tables are skipped. Start and end variables must be from the same `Original_table_name`.

2. **LLM 응답 파싱** — 로컬 exo LLM은 `timepoint_pairs = [(...), ...]` 형식으로 응답해야 함. `=` 기준으로 분리 후 `ast.literal_eval` 파싱. 형식 불일치 시 `ValueError` 발생. 응답이 마크다운 코드블록을 포함하면 파싱 실패할 수 있으니 system prompt의 output format 예시를 그대로 따르도록 설계됨.

3. **날짜 변환 실패** — `pd.to_datetime(..., errors='coerce')`로 변환 불가한 값은 NaT → `dropna()` 로 제외됨.

4. **원본 코드 개선 사항**:
   - `combined_time_df['Value'] = ...` SettingWithCopyWarning → `.copy()` 후 할당
   - LLM 응답 파싱 `try/except` 추가 (원본은 파싱 실패 시 unhandled exception)
   - `_validate_sequence` 내 `pd.concat` loop → list 수집 후 한 번에 concat
   - `os.path.join` 사용 (문자열 연결 대신)
   - `required=True` for `--config`

5. **Dependencies** — `pandas`, `numpy` (LLM: local exo LLM, OpenAI-compatible, timeout 180s)

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Sequence_Validity.py (이성민 작성)
