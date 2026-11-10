---
name: date-validity
description: Validate date values in a QUIQ-format table. Checks Event_date column and Mapping_info_1='date' rows using standard format parsing, Korean date formats, and optional LLM fallback. Use for data quality assessment of temporal data.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
  use_llm:
    description: "If true (default), uses the local exo LLM for ambiguous date fallback. Set false to skip LLM and use rule-based only."
    type: boolean
---

# Date Validity

Validates date values in a QUIQ-format table. For each date-related value, determines whether it represents a valid date using standard format parsing, Korean date format regex, and (optionally) an LLM fallback for ambiguous strings.

## When to Use This Skill

- After QUIQ conversion, to assess quality of temporal data
- To identify records with invalid or malformed date strings
- As part of LYDUS quality management assessment

## Data Sources in QUIQ

두 종류의 날짜 데이터를 대상으로 함:

| 소스 | 조건 | Variable_name |
|------|------|--------------|
| `Event_date` 컬럼 | non-null | `Event_date` |
| `Value` 컬럼 | `Mapping_info_1` contains `date` | 원본 Variable_name |

## Validation Pipeline

각 날짜 문자열에 대해 순서대로 검증:

```
1. dateutil.parse()          → 표준 날짜 형식 (ISO, US, EU 등)
2. _valid_date_custom()      → 한국어 날짜 (e.g. "2024년 3월 15일")
3. LLM fallback (선택)       → 위 두 방법 실패 시 로컬 exo LLM에 yes/no 질의
```

**SQL 버전** (`duckdb.sql`) 은 1번(TRY_STRPTIME 7종) + 2번(regex) 만 지원. LLM fallback 없음.

## Output

| File | Description |
|------|-------------|
| `date_validity_total.txt` | Overall Date Validity (%), Total/Invalid dates |
| `date_validity_summary.csv` | Per-variable: Total_date, Invalid_date, Date_Validity_(%) |
| `date_validity_detail.csv` | Per-row: Date_value, Is_valid (Python 버전만) |

## How to Run

### SQL 버전 (빠름, LLM 없음)

```python
import os
import duckdb

skill_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(skill_dir, "scripts/duckdb.sql")) as f:
    sql = f.read()

quiq_csv = "/path/to/quiq_3patients.csv"
sql = sql.replace("{quiq_csv}", quiq_csv)

df = duckdb.sql(sql).df()

total_date = df["Total_date"].sum()
invalid_date = df["Invalid_date"].sum()
date_validity = round((total_date - invalid_date) / total_date * 100, 2)
print(f"Date Validity (%) = {date_validity}")

save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/date_validity_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/date_validity_total.txt", "w") as f:
    f.write(f"Date Validity (%) = {date_validity}\n")
    f.write(f"Total dates = {total_date}\n")
    f.write(f"Invalid dates = {invalid_date}\n")
print(f"Saved {len(df):,} rows → {save_path}")
```

### Python 버전 (LLM fallback 포함)

```python
import pandas as pd
import sys, os
skill_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, skill_dir)
from scripts.date_validity import get_date_validity

quiq = pd.read_csv("/path/to/quiq.csv")
valid_results_df, summary_df = get_date_validity(
    quiq=quiq,
    use_llm=True   # False 로 설정하면 LLM fallback 비활성화
)
```

### As a script with config

```yaml
# config.yaml
quiq_path:  /path/to/quiq.csv
save_path:  /path/to/output
use_llm:    true   # false 로 설정하면 LLM fallback 없음
```

```bash
python scripts/date_validity.py --config config.yaml
```

## Critical Notes

1. **SQL vs Python 선택** — MIMIC-IV 처럼 ISO 형식 날짜만 있으면 SQL 버전으로 충분. 희귀 형식이나 자연어 날짜가 섞인 데이터는 Python + LLM fallback 권장.

2. **원본 코드 버그 수정** — `_gpt_chat` 반환값이 `list`인데 `if "no" in result`로 list 전체를 검색했음 → `result[0]`로 수정.

3. **LLM 호출 방식** — LLM fallback은 표준 파싱 실패 시에만 로컬 exo LLM (OpenAI 호환 /v1/chat/completions)를 통해 호출됨. MIMIC-IV에서는 거의 호출되지 않음.

4. **use_llm=False** — Python 버전에서 `use_llm=False` 로 설정하면 LLM fallback 없이 규칙 기반으로만 실행 가능.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Date_Validity.py (이성민 작성)
