---
name: note-fidelity
description: Evaluate completeness (fidelity) of unstructured clinical notes and radiology reports in a QUIQ-format table. Uses a local exo LLM to check whether required template items are mentioned in each note. Runs on an on-prem local exo LLM (closed-network safe; no external API key). Use for LYDUS data quality assessment of note_clinical and note_rad variables.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file. Must contain rows where Mapping_info_1 is 'note_clinical' or 'note_rad'.
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
---

# Note Fidelity

Evaluates the **completeness (fidelity)** of unstructured clinical notes and radiology reports by asking an LLM whether each required template item is present in the note. Covers clinical notes (`note_clinical`) and radiology reports (`note_rad`).

## When to Use This Skill

- After QUIQ conversion, to assess whether clinical notes contain all expected structured items
- To detect notes that are missing key sections (e.g., no "Diagnosis" in an admission note)
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** Requires the local exo LLM for note content analysis.

## Filtering Logic

| Type | `Mapping_info_1` | Supported `Mapping_info_2` codes |
|------|-----------------|----------------------------------|
| Clinical note | `note_clinical` | ADM, DIS, SUR, EME |
| Radiology report | `note_rad` | ACT, BCT, CCT, SCT, CXR, AXR, SXR, ECH |

Notes with unrecognized `Mapping_info_2` codes are skipped ("Invalid mapping or template missing").

## Report Templates

### Clinical Notes
| Code | Note Type | Template Items |
|------|-----------|---------------|
| ADM | Admission note | Admission department, Chief complaint, Present illness, Past medical history, Social & Family history, Physical examination, Review of systems, Diagnosis, Treatment plan |
| DIS | Discharge summary | Admission/Discharge department, Discharge reason, Diagnosis, Summary of progression, Medical prescription, Surgery or procedure, Treatment result, Discharge plan, Discharge form |
| SUR | Surgery note | Pre/Postoperative diagnosis, Pre/Postoperative procedure, Type of anesthesia, Operative findings, Surgical procedure |
| EME | Emergency note | Visit information, Past medical history, Medication history, Present illness, Examination findings, Presumptive diagnosis, Treatment plan |

### Radiology Reports
| Code | Modality | Template Items (examples) |
|------|----------|--------------------------|
| ACT | CT abdomen | Liver, Gallbladder, Spleen, Pancreas, Adrenals, Kidneys, Bowel, ... |
| BCT | CT brain | Extra-axial spaces, Ventricular system, Cerebral parenchyma, ... |
| CCT | CT chest | Pulmonary Parenchyma, Pleural Space, Heart and Pericardium, ... |
| SCT | CT spine | Alignment, Bones, Intervertebral Discs, Spinal canal, ... |
| CXR | X-ray chest | Lungs, Heart, Mediastinum, Pleural Spaces, Osseous Structures |
| AXR | X-ray abdomen | Bowel gas pattern, Abnormal calcifications, Bones, Others |
| SXR | X-ray spine | Alignment, Vertebral bodies, Intervertebral spaces, Soft Tissues |
| ECH | Echocardiography | Left/Right ventricle, Left/Right atrium, Aortic/Mitral/Tricuspid/Pulmonic valve, Pericardium, Aorta, ... |

## Scoring Logic

LLM response classifies each template item as `mentioned` or `not mentioned`.

**Fidelity Score (%)** = `(1 - not_mentioned / total_items) × 100`

- Clinical: items extracted via regex `:\s*("[^"]+"|[^:\n]+)`
- Radiology: items extracted by splitting on newlines

**Overall Note Fidelity** = unweighted mean across all scored notes.

## Output

| File | Description |
|------|-------------|
| `note_fidelity_total.txt` | Overall Note Fidelity (%) |
| `note_fidelity_summary.csv` | Per-(Mapping_info_1, Mapping_info_2): Count, mean, std fidelity |
| `note_fidelity_total_detail.csv` | Per-note: LLM response + fidelity score |
| `note_fidelity_plot.png` | Box plot of fidelity by note category |

## How to Run

```python
import pandas as pd
from scripts.note_fidelity import get_note_fidelity

quiq = pd.read_csv("/path/to/quiq.csv")

df_clinical, df_radiology, result_df, summary_df, \
mean_clinical, std_clinical, mean_radiology, std_radiology = get_note_fidelity(
    quiq=quiq
)

mean_fidelity = round(result_df['Fidelity_results'].mean(), 2)
print(f"Note Fidelity (%) = {mean_fidelity}")
print(f"Clinical: {mean_clinical} ± {std_clinical}")
print(f"Radiology: {mean_radiology} ± {std_radiology}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
```

```bash
python scripts/note_fidelity.py --config config.yaml
```

## Critical Notes

1. **Retry logic** — each LLM call retries up to 3 times on failure. Rows still failing after retries are recorded as "Failed after retries" and excluded from scoring.

2. **Invalid mapping** — notes with `Mapping_info_2` codes not in the template list return "Invalid mapping or template missing" and are excluded from scoring.

3. **Note Fidelity vs Note Accuracy** — Fidelity measures *what items are present* (template completeness); Accuracy measures *whether the content is correct* (error detection).

4. **원본 코드 개선 사항**:
   - `process_row` + `retry_process_row` inner 함수가 `_run_clinical` / `_run_radiology` 양쪽에 중복 → `_call_with_retry` 공통 함수로 통합
   - 오타 수정: `Present ilness` → `Present illness`, `Familty history` → `Family history`, `Physical exammination` → `Physical examination`
   - `get_unstructured_fidelity` → `get_note_fidelity` (스킬 이름과 일관성)
   - 템플릿/매핑 딕셔너리를 모듈 수준 상수로 분리

5. **Dependencies** — `pandas`, `seaborn`, `matplotlib`, `tqdm` (LLM: local exo LLM, OpenAI-compatible)

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Note_Fidelity.py (이성민 작성)
