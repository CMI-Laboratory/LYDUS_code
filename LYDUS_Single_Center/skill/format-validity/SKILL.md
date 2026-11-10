---
name: format-validity
description: Validate format of medical codes (ICD-9/10/11, SNOMED-CT, RxNorm, LOINC, ATC) in a QUIQ-format table. Identifies code type from variable name/description, then validates each value against the corresponding regex. Use for data quality assessment of medical code fields.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file (output of quiq skill).
    type: string
  via_path:
    description: Path to Variable Information Annotation CSV (columns: Original_table_name, Variable_name, Description).
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
  use_llm:
    description: "If true (default), uses the local exo LLM as fallback for unknown code types. Set false to use rule-based matching only."
    type: boolean
---

# Format Validity

Validates the **format** of medical codes in a QUIQ-format table. Targets rows where `Mapping_info_1 = 'medical_code'`, identifies the code system (e.g. ICD-10), and checks whether each value matches the expected format via regex.

## When to Use This Skill

- After QUIQ conversion, to verify that medical codes follow standard format rules
- To detect malformed ICD codes, LOINC codes, etc.
- As part of LYDUS quality management assessment

## Supported Code Types

| Code System | Regex Pattern | Detection Keywords |
|-------------|--------------|-------------------|
| ICD-9 | `^[0-9]{3}(\.[0-9]{1,2})?$` | icd + 9 |
| ICD-10 | `^[A-Z]{1}[0-9]{2}(\.[0-9]{1,2})?$` | icd + 10 |
| ICD-11 | `^[A-Z0-9][A-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,2})?$` | icd + 11 |
| SNOMED-CT | `^[0-9]{6,18}$` | snomed + ct |
| RxNorm | `^[0-9]{5,9}$` | rxnorm |
| LOINC | `^[0-9]{1,6}-[0-9]{1}$` | loinc |
| ATC | `^[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}$` | atc |

Unknown code types → `Is_valid = NULL` (SQL) or `Is_valid = False` (Python without LLM)

## Validation Pipeline

```
1. 규칙 기반 (match_code_regex)
   Variable_name + VIA Description 에서 키워드 탐지 → regex 반환

2. LLM fallback (선택, llm_define_regex)
   규칙으로 미분류 시 로컬 exo LLM에 코드명 + regex 질의

3. regex 검증
   각 Value에 str.match(regex) 적용
```

## Output

| File | Description |
|------|-------------|
| `format_validity_total.txt` | Overall Format Validity (%), Total/Invalid codes |
| `format_validity_summary.csv` | Per-variable: Total_code, Invalid_code, Format_Validity (%), Regular_Expression |
| `format_validity_detail.csv` | Per-row: Value, Is_valid |

## How to Run

### SQL 버전 (규칙 기반만, LLM 없음)

```python
import os
import duckdb

skill_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(skill_dir, "scripts/duckdb.sql")) as f:
    sql = f.read()

quiq_csv = "/path/to/quiq_3patients.csv"
via_csv  = "/path/to/via.csv"   # VIA 없으면 headers만 있는 빈 CSV 사용
sql = sql.replace("{quiq_csv}", quiq_csv)
sql = sql.replace("{via_csv}", via_csv)

df = duckdb.sql(sql).df()

total_code = df["Total_code"].sum()
invalid_code = df["Invalid_code"].sum()
format_validity = round((total_code - invalid_code) / total_code * 100, 2)
print(f"Format Validity (%) = {format_validity}")

save_path = "/path/to/output"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/format_validity_summary.csv", index=False, encoding="utf-8-sig")
with open(f"{save_path}/format_validity_total.txt", "w") as f:
    f.write(f"Format Validity (%) = {format_validity}\n")
    f.write(f"Total Code = {total_code}\n")
    f.write(f"Invalid Code = {invalid_code}\n")
print(f"Saved {len(df):,} rows → {save_path}")
```

### Python 버전 (LLM fallback 포함)

```yaml
# config.yaml
quiq_path:  /path/to/quiq.csv
via_path:   /path/to/via.csv
save_path:  /path/to/output
use_llm:    true   # false 로 설정하면 LLM fallback 없음
```

```bash
python scripts/format_validity.py --config config.yaml
```

## Critical Notes

1. **VIA 없이 SQL 실행** — `{via_csv}` 자리에 헤더만 있는 빈 CSV를 넣으면 Variable_name 기반 키워드 매칭만 동작함.
   ```
   Original_table_name,Variable_name,Description
   ```

2. **원본 코드 버그 수정** — 집계 시 `Invalid_code = sum(Is_valid)` (valid count) 로 계산한 뒤 나중에 `total - valid` 로 역산하는 혼란스러운 로직 → Python 버전에서 `Valid_code` / `Invalid_code` 명확히 분리.

3. **정수형 float 처리** — `4019.0` 같은 값은 `'4019'` 로 변환 후 검증 (SQL: regex 로 처리, Python: `str(int(x))`).

4. **Unknown 코드** — SQL 버전에서 7종 외 코드는 `Unknown_code` 컬럼으로 집계됨. LLM fallback 없이는 format validity 계산에서 제외됨.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Format_Validity.py (이성민 작성)
