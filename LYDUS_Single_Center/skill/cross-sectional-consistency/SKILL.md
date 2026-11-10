---
name: cross-sectional-consistency
description: Calculate cross-sectional consistency for categorical string variables in a QUIQ-format table using a local exo LLM to group semantically equivalent values. Runs on an on-prem local exo LLM (closed-network safe; no external API key). Use for data quality assessment when the same real-world concept may be expressed in multiple ways (e.g., 'F', 'Female', '여자').
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
---

# Cross-Sectional Consistency

Calculates **cross-sectional consistency** for categorical string variables in a QUIQ-format table. Uses a local exo LLM to group unique values by semantic equivalence, then measures how consistently a single canonical form is used within each semantic group.

## When to Use This Skill

- The same concept is expressed in multiple forms (e.g., `'F'`, `'Female'`, `'여자'`)
- To detect cross-institutional or cross-user naming inconsistencies
- As part of LYDUS quality management assessment

## Input Requirements

**QUIQ CSV** — standard QUIQ-format output. Variables analyzed must meet all of:
1. `Variable_type` contains `string` or `str`
2. `Mapping_info_1` does NOT contain `note`, `code`, or `date`
3. `Is_categorical == 1`
4. `Value` is not null
5. At least 2 unique values (variables with only 1 unique value are skipped)

**VIA CSV** (Variable Information Annotation) — provides natural-language descriptions per variable to help the LLM make accurate groupings.

| Column | Description |
|--------|-------------|
| `Original_table_name` | Table name (matches QUIQ) |
| `Variable_name` | Variable name (matches QUIQ) |
| `Description` | Natural language description of the variable |

## How It Works

For each `(Original_table_name, Variable_name)` group:
1. Unique values are sent to the LLM with a system prompt and variable description
2. LLM groups semantically equivalent values (e.g., `['F', 'Female', '여']` → one group)
3. Within each group, **inner consistency** is calculated:
   - For each semantic group, measure how dominant the most frequent form is
   - Score = `Σ (count²/total)` / `total` across all groups → closer to 1.0 = one canonical form dominates
4. LLM is retried up to 5 times if no values match

**Average Cross-Sectional Consistency** = unweighted mean across all valid variables.

## Output

| File | Description |
|------|-------------|
| `cross_sectional_consistency_total.txt` | Average Cross-Sectional Consistency (%) |
| `cross_sectional_consistency_summary.csv` | Per-variable consistency scores |
| `cross_sectional_consistency_detail.txt` | LLM-assigned semantic groups per variable |

## How to Run

```python
import pandas as pd
from scripts.cross_sectional_consistency import get_cross_sectional_consistency

quiq = pd.read_csv("/path/to/quiq.csv")
via  = pd.read_csv("/path/to/via.csv")

average_consistency, results_df, detail = get_cross_sectional_consistency(
    quiq=quiq,
    via=via
)

print(f"Average Cross-Sectional Consistency (%) = {round(average_consistency * 100, 2)}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
via_path:  /path/to/via.csv
save_path: /path/to/output
```

```bash
python scripts/cross_sectional_consistency.py --config config.yaml
```

## Critical Notes

1. **SQL 변환 불가** — 로컬 exo LLM 호출 및 응답 파싱이 핵심 로직이므로 DuckDB/BigQuery SQL로 표현할 수 없음.

2. **VIA 파일 필수** — `via_path`가 없으면 모든 변수에 `"No description available"`이 전달되어 LLM 정확도가 낮아짐.

3. **LLM 재시도 로직** — 응답이 실제 값과 매칭되지 않으면 최대 5회 재시도. 여전히 실패하면 해당 변수는 결과에서 제외됨.

4. **API 키 불필요** — LLM 호출은 사내 로컬 exo LLM (OpenAI 호환 /v1/chat/completions)를 통해 처리되므로 폐쇄망 안전하며 외부 API 키가 필요 없음.

5. **Is_categorical 타입** — 원본 코드에서 `Is_categorical == 1` 비교 전 `pd.to_numeric()` 변환을 누락한 버그가 있었음. 스킬 버전에서 수정됨.

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Cross_Sectional_Consistency.py (이성민 작성)
