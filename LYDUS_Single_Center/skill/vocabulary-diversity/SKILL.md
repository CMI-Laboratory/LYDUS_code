---
name: vocabulary-diversity
description: Measure diversity of noun words in clinical notes using NLTK POS tagging; computes unique/total noun ratio and top-5/10/20% coverage scores. No API key required. Use for LYDUS data quality assessment of vocabulary richness in unstructured text.
tier: community
category: lydus
parameters:
  quiq_path:
    description: Path to QUIQ-format CSV file. Must contain note rows (Mapping_info_1 contains 'note').
    type: string
  save_path:
    description: Directory path to save output files and plots.
    type: string
  top_n:
    description: Number of top words to show in the histogram plot. Default 10.
    type: integer
---

# Vocabulary Diversity

Measures the **diversity of noun words** in clinical notes extracted from a QUIQ-format table. High diversity means notes use varied vocabulary; low diversity means a small set of nouns dominates all documentation.

## When to Use This Skill

- After QUIQ conversion, to assess how varied the noun vocabulary is in clinical notes
- To detect template-heavy or formulaic documentation (low vocabulary diversity)
- To identify the most frequently used clinical terms (coverage score analysis)
- As part of LYDUS quality management assessment

## SQL Support

**Not applicable.** Requires NLTK word tokenization and POS tagging.

## Filtering Logic

| Condition | Value |
|-----------|-------|
| `Mapping_info_1` | contains `note` (case-insensitive) — covers `note_clinical`, `note_rad`, etc. |
| `Value` | used as the note text |

## Algorithm

1. **Word tokenization** (`_noun_words`): tokenize each note with NLTK `word_tokenize`
2. **Noun filter**: keep only tokens with POS tag starting with `NN` (noun, proper noun, etc.)
3. **Diversity metrics**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `Vocabulary Diversity (%)` | `unique_nouns / total_nouns × 100` | 높을수록 다양한 명사 어휘 사용 |
| `Coverage Score (Top X%)` | Top X% of unique noun types → % of total occurrences | 상위 X% 어휘가 전체 명사의 몇 % 차지 |

**Coverage score example**: Top 5% = the most-common 5% of unique noun types cover Y% of all noun occurrences. Low Y% → diverse vocabulary; high Y% → a few terms dominate.

## Comparison with sentence-diversity

| Skill | Unit of analysis | POS filter |
|-------|-----------------|-----------|
| sentence-diversity | sentences | VB* (verbs) |
| vocabulary-diversity | words | NN* (nouns) |

## NLTK Dependencies

The script auto-downloads required NLTK data on first run:
- `punkt` / `punkt_tab` — word tokenizer
- `averaged_perceptron_tagger` / `averaged_perceptron_tagger_eng` — POS tagger

## Output

| File | Description |
|------|-------------|
| `vocabulary_diversity_summary.txt` | Vocabulary Diversity (%), Coverage Scores for top 5/10/20% |
| `vocabulary_diversity_frequency.csv` | Per-word: Count, Percentage (sorted by frequency) |
| `vocabulary_diversity_plot.png` | Bar chart of top-N most frequent nouns |

## How to Run

```python
import pandas as pd
from scripts.vocabulary_diversity import get_vocabulary_diversity

quiq = pd.read_csv("/path/to/quiq.csv")

word_diversity, item_vs_pct, coverage_scores, freq_df = get_vocabulary_diversity(
    quiq=quiq,
    top_n=10
)

print(f"Vocabulary Diversity (%) = {word_diversity}")
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
python scripts/vocabulary_diversity.py --config config.yaml
```

## Critical Notes

1. **note 필터만 적용** — `Mapping_info_1`에 `note`가 포함된 모든 행이 대상 (`note_clinical`, `note_rad` 등).

2. **Noun 단어만 집계** — NN* POS 태그만 포함. 동사, 형용사, 관사 등은 제외. 임상 개체명(진단명, 약물명 등)이 주요 분석 대상.

3. **NLTK 자동 다운로드** — 첫 실행 시 punkt, averaged_perceptron_tagger가 없으면 자동으로 다운로드됨.

4. **원본 코드 개선 사항**:
   - `df_note['TEXT_words'] = ...` SettingWithCopyWarning → `.copy()` 후 할당
   - `assert len(df_note) > 0` → 명시적 FAIL 메시지 후 빈 결과 반환 (AssertionError 방지)
   - `total_words` 변수 이중 할당 혼란 → `total_count` / `counter` 명확히 분리
   - `os.path.join` 사용 (f-string 경로 대신)
   - `--config` argparse에 `required=True` 추가
   - `matplotlib.use('Agg')` 추가 (헤드리스 환경)

5. **Dependencies** — `nltk`, `pandas`, `matplotlib`

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 (비공개 내부 문서)
- Original Python implementation: LYDUS_Vocabulary_Diversity.py (이성민 작성)
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.
