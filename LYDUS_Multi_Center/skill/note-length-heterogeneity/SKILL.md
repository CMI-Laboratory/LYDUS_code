---
name: note-length-heterogeneity
description: Compare the text-length distributions of clinical notes and radiology reports between two hospitals (LYDUS multi-center B-2). Computes the Wasserstein distance between character-length distributions per note type and normalizes it by the pooled median length. Input is two QUIQ-format CSVs. No API key required. Use for multi-center unstructured-text heterogeneity, e.g. to see whether one site writes much longer or shorter notes than another.
tier: community
category: lydus
parameters:
  q1:
    description: Path to the QUIQ-format CSV for site 1.
    type: string
  q2:
    description: Path to the QUIQ-format CSV for site 2.
    type: string
  out_b2:
    description: Output CSV path for the B-2 result.
    type: string
  min_n_wd:
    description: Minimum notes each site needs for a note type to be compared (default 10).
    type: integer
---

# Note Length Heterogeneity (B-2)

Measures how different the **lengths** of notes are between two hospitals, per note type. Large differences usually point to different templates, documentation habits or truncation, and can bias models trained on pooled notes.

## When to Use This Skill

- Before pooling clinical notes or radiology reports from two centers
- To check whether note length differs by site for the same note type
- As part of a LYDUS multi-center quality assessment (pair with `note-style-separability`; for structured data see `multi-structured-consistency`)

## Script

`/Users/leesungmin/Desktop/graduate_school/07_lydus_skills/lydus_multi_skills/LYDUS_Multi_Unstructured_2.py`

Template config: `config_multi_unstructured_2.yml` in the same folder.

## How to Run

1. Ask the user for `q1`, `q2` and an output path if not given; use `min_n_wd: 10` unless they say otherwise.
2. Write a config YAML (do not overwrite the template):

```yaml
q1: /path/to/quiq_site1.csv
q2: /path/to/quiq_site2.csv
out_b2: /path/to/b2_result.csv
min_n_wd: 10
```

3. Run it, always passing `--config` (the script's default is `config.yaml` in the current directory):

```bash
python /Users/leesungmin/Desktop/graduate_school/07_lydus_skills/lydus_multi_skills/LYDUS_Multi_Unstructured_2.py --config /path/to/config.yml
```

Dependencies: `pandas numpy scipy pyyaml`.

## Required Input Columns

| Column | Use |
|--------|-----|
| `Recorder_affiliation` | Site label. Must have exactly **one** unique non-null value per file; otherwise the site becomes `unknown`. |
| `Mapping_info_1` | Rows with `note_clinical` or `note_rad` are used |
| `Mapping_info_2` | Note subtype; `Note_type = Mapping_info_1 + '_' + Mapping_info_2` |
| `Value` | Note text (null rows dropped) |

Check `Recorder_affiliation` before running: if both files end up as `unknown`, or share the same value, there is only one site and nothing is compared.

## Method

Length = number of **characters** in the note (`str.len()`, not tokens despite the `Token_count` name).

For each `Note_type` with ≥ 2 sites, for each site pair:

1. Skip the pair if either site has fewer than `min_n_wd` notes.
2. `WD` = Wasserstein distance between the two length distributions (in characters) — roughly the average number of characters by which the two distributions differ.
3. `WD_normalized = WD / median(pooled lengths of both sites)`.

`WD_normalized` is unitless: 0.10 means the sites differ by about 10% of a typical note's length. Lower = more similar.

## Output

CSV at `out_b2` (UTF-8 with BOM), one row per note type × site pair:

| Column | Meaning |
|--------|---------|
| `Note_type` | e.g. `note_clinical_discharge` |
| `Site_A`, `Site_B` | Sites compared |
| `N_A`, `N_B` | Number of notes |
| `Median_A`, `Median_B` | Median length (characters) |
| `Wasserstein_distance` | WD in characters |
| `Median_pooled` | Median length of both sites combined |
| `WD_normalized` | `WD / Median_pooled` |

The console log also prints min / median / mean / max length per site.

## Reporting Back

For each note type, give both medians, the WD and `WD_normalized` as a percentage, and say which site writes longer notes. List note types or pairs that were skipped for too few notes.
