---
name: multi-structured-consistency
description: Compare structured clinical data from two hospitals (two QUIQ-format tables, A and B) and compute 7 LYDUS multi-center quality metrics - variable name consistency, categorical value consistency, categorical distribution homogeneity (JSD), continuous distribution homogeneity (normalized Wasserstein), continuous distribution shape consistency (skew/kurtosis/dip), measurement unit consistency, and medical code system consistency. Uses the OpenAI API to match variables and group values. Use when assessing cross-site harmonization or heterogeneity of structured EHR data between two centers.
tier: community
category: lydus
parameters:
  quiq_a_path:
    description: Path to the QUIQ-format CSV for hospital A.
    type: string
  quiq_b_path:
    description: Path to the QUIQ-format CSV for hospital B.
    type: string
  via_a_path:
    description: Path to the VIA (variable description) CSV for hospital A. Used only by metric 07.
    type: string
  via_b_path:
    description: Path to the VIA CSV for hospital B. Used only by metric 07.
    type: string
  model_ver:
    description: OpenAI model name used for variable matching, value clustering and code-system inference.
    type: string
  api_key:
    description: OpenAI API key.
    type: string
  save_path:
    description: Existing directory where result files are written.
    type: string
---

# Multi-Center Structured Data Consistency

Compares the structured data of **two hospitals (A, B)**, both already converted to QUIQ format, and measures how consistent they are across 7 metrics. Run it in one go: metrics 02–06 all depend on the variable pairs found in 01.

## When to Use This Skill

- Before pooling or federating structured EHR data from two centers
- To find which variables differ in naming, value encoding, units, distributions or code systems between sites
- As the structured-data part of a LYDUS multi-center quality assessment (for notes, see `note-style-separability` and `note-length-heterogeneity`)

## Script

`/Users/leesungmin/Desktop/graduate_school/07_lydus_skills/lydus_multi_skills/LYDUS_Multi_Structured.py`

Template config: `config_multi_structured.yml` in the same folder.

## How to Run

1. Ask the user for the paths in **Parameters** if they were not given. Make sure `save_path` exists (create it if needed).
2. Ask for the OpenAI API key and model name. Do not print the key back in chat.
3. Write a config YAML (do not overwrite the template), for example in `save_path`:

```yaml
quiq_a_path: /path/to/quiq_a.csv
quiq_b_path: /path/to/quiq_b.csv
via_a_path: /path/to/via_a.csv
via_b_path: /path/to/via_b.csv
model_ver: <openai-model>
api_key: <key>
save_path: /path/to/results
```

4. Run (`--config` is required; there is no default):

```bash
python /Users/leesungmin/Desktop/graduate_school/07_lydus_skills/lydus_multi_skills/LYDUS_Multi_Structured.py --config /path/to/config.yml
```

Dependencies: `pandas numpy scipy openai pyyaml diptest`.

It calls the LLM many times (per chunk of 50 variables, per matched categorical pair, per medical-code variable), so it can take a while on large tables. Run it in the background if needed.

## Required Input Columns

| File | Columns used |
|------|--------------|
| QUIQ A/B | `Variable_name`, `Value`, `Is_categorical` (1/0), `Unit`, `Mapping_info_1`, `Original_table_name` |
| VIA A/B | `Original_table_name`, `Variable_name`, `Description` |

## Pipeline

Variable names are lower-cased and stripped before matching. Three subsets are made from each table:

| Subset | Filter | Used by |
|--------|--------|---------|
| categorical | `Is_categorical == 1` | 01, 02, 03 |
| continuous | `Is_categorical == 0` | 01, 04, 05 |
| unit | `Unit` not empty | 01, 06 |
| medical code | `Mapping_info_1 == 'medical_code'` | 07 |

A metric is skipped (no file written) when its subset has no matched pairs or is empty.

### 01 Variable Name Consistency
The LLM matches A variables to B variables that mean the same clinical concept (JSON pairs with a score); pairs with score ≥ 0.8 are kept. A pair is *heterogeneous* when the two names differ.
`Consistency (%) = (1 − mean(Is_hetero)) × 100`

### 02 Categorical Value Consistency
For each matched categorical pair, the LLM groups the union of A and B values into clusters of the same meaning (e.g. `M` / `Male`). A cluster is a mismatch when A and B use different surface strings for it.
`surface_match_rate (%) = (1 − n_mismatch / n_clusters) × 100`, averaged over pairs.

### 03 Categorical Variable Distribution Homogeneity
Values are mapped to the clusters from 02; Jensen–Shannon divergence is computed over clusters present in both sites (needs ≥ 2 shared clusters, otherwise NaN). Reported as mean and std of JSD; **lower = more similar** (0 to ln 2).

### 04 Continuous Variable Distribution Homogeneity
Wasserstein distance between A and B numeric values, divided by the std of the pooled values. Mean and std reported; **lower = more similar**.

### 05 Continuous Variable Distribution Shape Consistency
For each site (needs ≥ 20 values): skewness, kurtosis and Hartigan's dip statistic.
`shape_distance = |Δskew| + |Δkurtosis| + |Δdip|`. Mean and std reported; **lower = more similar**.

### 06 Measurement Unit Consistency
For each matched pair, the union of lower-cased units across both sites. A pair is *mixed* when the union holds more than one unit.
`Consistency (%) = (1 − mean(is_mixed)) × 100`

### 07 Medical Code Consistency
For every `(Original_table_name, Variable_name)` with `Mapping_info_1 == 'medical_code'`, the LLM infers a semantic type (DIAGNOSIS / PROCEDURE / LAB / DRUG / OTHER) and the code systems (ICD-9, ICD-10, CPT, LOINC, RXNORM, ATC, SNOMED, …) from the table name, variable name, VIA description and 50 sampled values. Code systems are pooled per semantic type for each site; a semantic type is a mismatch when the two sets differ.
`Consistency (%) = (1 − mean(Is_mismatch)) × 100`

## Output (in `save_path`)

| File | Content |
|------|---------|
| `LYDUS_multi_structured_total_results.txt` | Summary of all computed metrics |
| `01_Variable_name_consistency.csv` | `A_var_name`, `B_var_name`, `Is_hetero` |
| `02_Categorical_value_consistency.csv` | Per pair: `surface_match_rate`, `n_clusters`, `n_mismatch`, `cluster_detail` |
| `03_Categorical_value_distribution_homogeneity.csv` | Per pair: `A_dist`, `B_dist`, `js_divergence` |
| `04_Continuous_variable_distribution_homogeniety.csv` | Per pair: `wasserstein_dist`, `wd_normalized` |
| `05_Continuous_variable_distribution_shape_consistency.csv` | Per pair: skew / kurtosis / dip for A and B, `shape_distance` |
| `06_Measurement_unit_consistency.csv` | Per pair: unit sets and counts, `is_mixed` |
| `07_Medical_code_consistency_detail.csv` | Per variable: inferred semantic type and code systems |
| `07_Medical_code_consistency.csv` | Per semantic type: A/B code-system sets, `Is_mismatch` |

## Reporting Back

Read `LYDUS_multi_structured_total_results.txt` and give the user each metric with its direction (% metrics: higher is better; JSD / WD / shape distance: lower is better). Point out the worst pairs from the CSVs (e.g. lowest `surface_match_rate`, highest `js_divergence` or `wd_normalized`, mixed units, mismatched code systems). Say which metrics were skipped and why.

## Caveats

- LLM output is not deterministic: variable pairs and value clusters can change between runs, which changes 02–06.
- Metric 01 is only over *matched* pairs; unmatched variables do not lower the score.
- In 02, the variable name passed to the LLM prompt is the literal string `var_A<B name>` (an f-string typo in the script), so the LLM gets less context than intended.
- Sends variable names, category values and sampled codes to the OpenAI API. Confirm with the user that this is allowed for their data.
