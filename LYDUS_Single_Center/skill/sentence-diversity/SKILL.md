---
name: sentence-diversity
description: Measure diversity of verb-containing sentences extracted from clinical notes (Mapping_info_1 contains 'note') in a QUIQ-format table. Uses NLTK POS tagging to identify sentences with verbs, then computes unique/total ratio and coverage scores. No API key required. Use for LYDUS data quality assessment of note text diversity.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file. Must contain rows where Mapping_info_1 contains 'note'.
    type: string
  save_path:
    description: Directory path to save output files.
    type: string
  top_n:
    description: Number of top sentences to show in histogram (default 10).
    type: integer
---

# Sentence Diversity

Measures the **diversity of verb-containing sentences** in clinical notes extracted from a QUIQ-format table. High diversity means notes use varied phrasing; low diversity means many sentences are copy-pasted or templated.

## When to Use This Skill

- After QUIQ conversion, to assess how varied the language is in clinical notes
- To detect template-heavy documentation (low sentence diversity)
- To identify the most frequently repeated sentences (coverage score analysis)
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** Requires NLTK sentence tokenization and POS tagging.

## Filtering Logic

| Condition | Value |
|-----------|-------|
| `Mapping_info_1` | contains `note` (case-insensitive) — covers `note_clinical`, `note_rad`, etc. |
| `Value` | used as the note text |

## Algorithm

1. **Sentence segmentation** (`_custom_sent_tokenize`): split on 2+ consecutive newlines or non-word characters
2. **Verb filter** (`_verb_sentences`): keep only sentences that contain at least one word with POS tag starting with `VB` (verb)
3. **Diversity metrics**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `Sentence Diversity (%)` | `unique_sentences / total_sentences × 100` | 높을수록 다양한 표현 |
| `Coverage Score (Top X%)` | Top X% of unique sentence types → % of total | 상위 X% 표현이 전체의 몇 % 차지 |

**Coverage score example**: Top 5% = the most-common 5% of unique sentence types cover Y% of all sentence occurrences. Low Y% → diverse; high Y% → a few sentences dominate.

## NLTK Dependencies

The script auto-downloads required NLTK data on first run:
- `punkt` / `punkt_tab` — sentence/word tokenizer
- `averaged_perceptron_tagger` / `averaged_perceptron_tagger_eng` — POS tagger

## Output

| File | Description |
|------|-------------|
| `sentence_diversity_summary.txt` | Sentence Diversity (%), Coverage Scores for top 5/10/20% |
| `sentence_diversity_frequency.csv` | Per-sentence: Count, Percentage (sorted by frequency) |
| `sentence_diversity_plot.png` | Bar chart of top-N most frequent sentences |

## How to Run

```python
import pandas as pd
from scripts.sentence_diversity import get_sentence_diversity

quiq = pd.read_csv("/path/to/quiq.csv")

sen_diversity, item_vs_percentage, coverage_scores, freq_df = get_sentence_diversity(
    quiq=quiq,
    top_n=10
)

print(f"Sentence Diversity (%) = {sen_diversity}")
for k, v in coverage_scores.items():
    print(f"  Top {k}%: {v}")
```

### As a script with config

```yaml
# config.yaml
quiq_path: /path/to/quiq.csv
save_path: /path/to/output
top_n:     10   # optional, default 10
```

```bash
python scripts/sentence_diversity.py --config config.yaml
```

## Critical Notes

1. **note 필터만 적용** — `Mapping_info_1`에 `note`가 포함된 모든 행이 대상 (`note_clinical`, `note_rad` 등).

2. **Verb 문장만 집계** — 동사가 없는 단편적 표현(표 항목, 수치 나열 등)은 제외. 주로 서술형 문장만 분석.

3. **NLTK 자동 다운로드** — 첫 실행 시 punkt, averaged_perceptron_tagger가 없으면 자동으로 다운로드됨.

4. **top_n vs. item_vs_percentage** — `item_vs_percentage`는 모든 unique 문장을 빈도 순으로 포함 (histogram 용). `top_n`은 시각화에서 몇 개만 표시할지 결정.

5. **원본 코드 개선 사항**:
   - `df_note['TEXT_Verb'] = ...` SettingWithCopyWarning → `.copy()` 후 할당
   - `_verb_sentence` → `_verb_sentences` (복수형, 반환값 명확)
   - NLTK 데이터 자동 다운로드 로직 추가
   - `os.path.join` 사용 (문자열 연결 대신)
   - `--config` argparse에 `required=True` 추가

6. **Dependencies** — `nltk`, `pandas`, `matplotlib`

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Sentence_Diversity.py (이성민 작성)
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.
