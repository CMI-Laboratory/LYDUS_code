---
name: note-accuracy
description: Evaluate accuracy of unstructured clinical notes and radiology reports in a QUIQ-format table using a local exo LLM. Detects diagnostic, procedural, drug, demographic, and date errors in clinical notes; identifies critical errors in radiology impressions. Runs on an on-prem local exo LLM (closed-network safe; no external API key). Use for LYDUS data quality assessment of note_clinical and note_rad variables.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill). Must contain rows where Mapping_info_1 is 'note_clinical' or 'note_rad'.
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
---

# Note Accuracy

Evaluates the **accuracy of unstructured clinical notes and radiology reports** by sending each note to the local exo LLM for error detection. Covers clinical notes (`note_clinical`) and radiology reports (`note_rad`).

## When to Use This Skill

- After QUIQ conversion, to assess quality of free-text clinical documentation
- To detect clinically significant errors in admission/discharge notes, surgery notes, radiology impressions, etc.
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** Requires the local exo LLM for note review.

## Filtering Logic

| Type | `Mapping_info_1` | Description |
|------|-----------------|-------------|
| Clinical note | `note_clinical` | Admission, discharge, surgery, emergency notes |
| Radiology report | `note_rad` | CT, X-ray, echocardiography impression sections |

## Mapping_info_2 Code Reference

| Code | Note Type |
|------|-----------|
| ACT | CT abdomen |
| BCT | CT brain |
| CCT | CT chest |
| SCT | CT spine |
| CXR | X-ray chest |
| AXR | X-ray abdomen |
| SXR | X-ray spine |
| ECH | Echocardiography |
| ADM | Admission note |
| DIS | Discharge summary |
| SUR | Surgery note |
| EME | Emergency note |

## Evaluation Logic

### Clinical Notes (`note_clinical`)
LLM checks 6 error categories per note:
1. Spelling or grammatical error
2. **Diagnostic Information Error** (counted)
3. Drug Information Error
4. **Procedure Information Error** (counted)
5. Demographic Information Error
6. Date Information Error

**Score** = `(# of "No" responses among Diagnostic + Procedure) / 2 × 100`
- 100% = both categories error-free
- 50% = one of two has an error
- 0% = both have errors

### Radiology Reports (`note_rad`)
LLM identifies one critical error in the Impression section.

**Score** = `100` if no error, `0` if error found.

### Overall Note Accuracy
Unweighted mean across all notes (clinical + radiology combined).

## Output

| File | Description |
|------|-------------|
| `note_accuracy_total.txt` | Overall Note Accuracy (%) |
| `note_accuracy_summary.csv` | Per-(Mapping_info_1, Mapping_info_2) accuracy summary |
| `note_accuracy_total_detail.csv` | Per-note: LLM response + accuracy score |
| `note_accuracy_plot.png` | Box plot of accuracy by note category |

## How to Run

```python
import pandas as pd
from scripts.note_accuracy import get_note_accuracy

quiq = pd.read_csv("/path/to/quiq.csv")

df_clinical, df_radiology, result_df, summary_df = get_note_accuracy(
    quiq=quiq
)

mean_accuracy = round(result_df['Accuracy_results'].mean(), 2)
print(f"Note Accuracy (%) = {mean_accuracy}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/note_accuracy.py --config config.yaml
```

## Critical Notes

1. **Call with timeout** — each local exo LLM call (OpenAI-compatible /v1/chat/completions) runs with a 60-second timeout. Failed rows return an empty result.

2. **Clinical score logic** — only Diagnostic + Procedure categories count toward the clinical note score (not spelling, drug, demographic, date). This matches the original LYDUS specification.

3. **Radiology score parsing** — the LLM response is parsed as a Python dict using `ast.literal_eval`. Responses that cannot be parsed return `None` (excluded from average).

4. **원본 코드 개선 사항**:
   - `api_call(results=[])` 가변 기본인자 제거 → `_call_api_threaded` 함수로 리팩터링
   - `_run_clinical`과 `_run_radiology`에 중복된 `api_call`/`process_rows` 내부 함수 → `_process_notes` 공통 함수로 통합
   - `get_unstructured_accuracy` → `get_note_accuracy` 로 명칭 변경 (스킬 이름과 일관성)

5. **Dependencies** — `pandas`, `seaborn`, `matplotlib`, `tqdm` (LLM: local exo LLM, OpenAI-compatible)

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Note_Accuracy.py (이성민 작성)
