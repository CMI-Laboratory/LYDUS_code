---
name: via
description: Generate VIA (Variable Information Archive) table from a user-provided documentation file (PDF or text). Requires a data dictionary or schema documentation to produce meaningful variable descriptions. If no document is provided, the skill will ask for one before proceeding. Output is used as input to format-validity, cross-sectional-consistency, and bias-detection skills.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  doc_path:
    description: "Path to documentation file describing the variables (PDF, txt, md, etc.). REQUIRED — the skill will not run without this."
    type: string
  save_path:
    description: Directory path to save output file (via.csv).
    type: string
---

# VIA (Variable Information Archive)

Generates a **VIA table** by reading a user-provided documentation file and using an LLM to extract a description for every unique `(Original_table_name, Variable_name)` pair in the QUIQ dataset.

> ⚠️ **문서 필수**: VIA는 변수 설명 문서 없이 실행되지 않습니다. 데이터 딕셔너리, 코드북, 스키마 문서 등을 먼저 준비해주세요.

## When to Use This Skill

- Before running `format-validity` — VIA descriptions help identify the correct medical code system (ICD-9/10, LOINC, etc.)
- Before running `cross-sectional-consistency` — VIA provides context for LLM-based categorical value grouping
- Before running `bias-detection` — orchestrates both above skills

## VIA Table Schema

| Column | Type | Description |
|--------|------|-------------|
| `Original_table_name` | String | Source table name (same as in QUIQ) |
| `Variable_name` | String | Variable name (same as in QUIQ) |
| `Description` | String | Description extracted from the provided documentation |

## Supported Document Formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Plain text | `.txt` |
| Markdown | `.md` |
| CSV / TSV (data dictionary 형태) | `.csv`, `.tsv` |
| 기타 텍스트 파일 | 모두 가능 |

## Generation Pipeline

```
1. doc_path 미제공 시 → "문서가 필요합니다" 메시지 출력 후 종료
2. 문서 읽기 (PDF: pdfplumber, 텍스트: UTF-8)
3. QUIQ에서 (Original_table_name, Variable_name) 고유 조합 추출
4. 30개 단위 배치로 LLM에 질의:
   "이 문서에서 다음 변수들의 설명을 찾아 1-2문장으로 작성해줘"
5. 응답 파싱 → via.csv 저장
```

## Output

| File | Description |
|------|-------------|
| `via.csv` | VIA table: Original_table_name, Variable_name, Description |

## How to Run

```python
import pandas as pd
from scripts.via import get_via

quiq = pd.read_csv("/path/to/quiq.csv")
via_df = get_via(quiq, doc_path="/path/to/mimic_data_dictionary.pdf")
via_df.to_csv("/path/to/output/via.csv", index=False, encoding="utf-8-sig")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
doc_path:  /path/to/data_dictionary.pdf   # 필수
save_path: /path/to/output
```

```bash
python scripts/via.py --config config.yaml
```

## Critical Notes

1. **문서 미제공 시 실행 불가** — `doc_path`가 없거나 파일이 존재하지 않으면 스킬이 즉시 종료되며 문서를 요청하는 메시지를 출력합니다.

2. **배치 처리** — 변수 30개씩 묶어서 LLM에 한 번에 질의합니다. 전체 변수 수가 N이면 약 ⌈N/30⌉번 API 호출이 발생합니다.

3. **문서에 없는 변수** — 문서에서 찾지 못한 변수는 `"No description available in the provided document."` 로 채워집니다. 이 경우 downstream LLM 스킬은 변수명만으로 판단합니다.

4. **PDF 의존성** — PDF 읽기에 `pdfplumber` 사용. 미설치 시: `pip install pdfplumber`

5. **Consumed by downstream skills**:
   - `format-validity`: Description으로 의료 코드 체계 식별 (ICD-9/LOINC/etc.)
   - `cross-sectional-consistency`: 카테고리 값 시맨틱 그루핑 컨텍스트로 활용
   - `bias-detection`: 위 두 스킬에 VIA 전달

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- VIA 개념 정의: 스크린샷 기반 설계 (이성민, 2026-06-21)
