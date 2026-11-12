---
name: note-style-separability
description: Measure how distinguishable two hospitals' clinical notes and radiology reports are by writing style (LYDUS multi-center B-1). Trains a TF-IDF (1-2 gram) + logistic regression classifier to predict the site from note text and reports k-fold CV accuracy per note type; high accuracy (≥ 0.85) flags site-leakage risk. Input is two QUIQ-format CSVs. No API key required. Use for multi-center unstructured-text heterogeneity or site-leakage checks before pooling notes.
tier: community
category: lydus
parameters:
  q1:
    description: Path to the QUIQ-format CSV for site 1.
    type: string
  q2:
    description: Path to the QUIQ-format CSV for site 2.
    type: string
  out:
    description: Output CSV path for the B-1 result.
    type: string
  sample:
    description: Maximum notes sampled per site per note type (default 200).
    type: integer
  min_n:
    description: Minimum notes a site needs for a note type to be included (default 20).
    type: integer
---

# Note Style Separability (B-1)

Checks whether a simple text classifier can tell **which hospital wrote a note**. If it can, notes carry strong site-specific style (templates, abbreviations, phrasing), and a model trained on pooled notes may learn the site instead of the clinical content.

## When to Use This Skill

- Before pooling clinical notes or radiology reports from two centers
- To quantify site-specific writing style per note type
- As part of a LYDUS multi-center quality assessment (pair with `note-length-heterogeneity`; for structured data see `multi-structured-consistency`)

## Script

`/Users/leesungmin/Desktop/graduate_school/07_lydus_skills/lydus_multi_skills/LYDUS_Multi_Unstructured_1.py`

Template config: `config_multi_unstructured_1.yml` in the same folder.

## How to Run

1. Ask the user for `q1`, `q2` and an output path if not given; use the defaults for `sample` (200) and `min_n` (20) unless they say otherwise.
2. Write a config YAML (do not overwrite the template):

```yaml
q1: /path/to/quiq_site1.csv
q2: /path/to/quiq_site2.csv
out: /path/to/b1_result.csv
sample: 200
min_n: 20
```

3. Run it, always passing `--config` (the script's default is `config.yaml` in the current directory):

```bash
python /Users/leesungmin/Desktop/graduate_school/07_lydus_skills/lydus_multi_skills/LYDUS_Multi_Unstructured_1.py --config /path/to/config.yml
```

Dependencies: `pandas numpy scikit-learn pyyaml`.

## Required Input Columns

| Column | Use |
|--------|-----|
| `Recorder_affiliation` | Site label. Must have exactly **one** unique non-null value per file; otherwise the site becomes `unknown`. |
| `Mapping_info_1` | Rows with `note_clinical` or `note_rad` are used |
| `Mapping_info_2` | Note subtype; `Note_type = Mapping_info_1 + '_' + Mapping_info_2` |
| `Value` | Note text (null rows dropped) |

Check `Recorder_affiliation` before running: if both files end up as `unknown`, or share the same value, there is only one site and every note type is skipped.

## Method

For each `Note_type`:

1. Keep sites with ≥ `min_n` notes; skip the type if fewer than 2 sites remain.
2. Randomly sample up to `sample` notes per site (`random_state=42`).
3. TF-IDF: `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`.
4. Logistic regression (`C=1.0`), stratified k-fold CV with `k = min(5, smallest site count)`, scored by accuracy.

`Site_leakage_risk = CV_accuracy_mean ≥ 0.85`

With two balanced sites, chance accuracy is about 0.5. Values near 0.5 mean the styles are hard to tell apart; values near 1.0 mean the site is almost fully predictable from the text.

## Output

CSV at `out` (UTF-8 with BOM), one row per analysed note type:

| Column | Meaning |
|--------|---------|
| `Note_type` | e.g. `note_clinical_discharge`, `note_rad_ct` |
| `Sites` | Sites compared, `A / B` |
| `CV_accuracy_mean` | Mean CV accuracy |
| `CV_accuracy_std` | Std of fold accuracies |
| `Site_leakage_risk` | `True` if mean accuracy ≥ 0.85 |

The console log also prints per-site counts and lengths, the TF-IDF matrix size and per-fold accuracy.

## Reporting Back

List each note type with its mean ± std accuracy and the leakage flag, and say which types were skipped (too few notes or sites). Remind the user that sample sizes are capped by `sample`, so the result reflects style rather than volume.
