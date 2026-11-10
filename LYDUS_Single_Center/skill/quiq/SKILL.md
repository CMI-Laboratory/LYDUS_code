---
name: quiq
description: Convert MIMIC-IV clinical data into QUIQ (Quality Intelligence Unified Query) long-format table. Use when transforming MIMIC-IV tables to QUIQ format for data quality assessment, when applying QUIQ Mapping_info rules to MIMIC-IV variables, or when building a standardized long-format dataset from MIMIC-IV for quality management programs.
tier: community
category: lydus
datasets:
  - mimic-iv
parameters:
  bq_project:
    description: BigQuery project ID where MIMIC-IV data is stored. Varies by institution — PhysioNet public access uses 'physionet-data'; institutions with their own copy use their own project ID (e.g. 'cmi-lab').
    default: physionet-data
    type: string
---

# MIMIC-IV → QUIQ Format Conversion

Converts all MIMIC-IV tables (hosp, icu, ed schemas) into the QUIQ long-format table used by the LYDUS quality management program. Each original row becomes one or more QUIQ rows depending on table type (wide, code, or event).

## When to Use This Skill

- User asks to convert MIMIC-IV data to QUIQ format
- User wants to apply QUIQ Mapping_info rules to MIMIC-IV variables
- User needs a standardized long-format output for data quality assessment
- User references 가이드라인 제4권 §1.5 or QUIQ table structure

## QUIQ Table Schema

| Column | Description |
|--------|-------------|
| `Primary_key` | Row identifier (shared across rows from same original record) |
| `Variable_ID` | itemid for event tables, empty otherwise |
| `Original_table_name` | Source table (e.g., `ADMISSIONS`, `LABEVENTS`) |
| `Variable_name` | Column name or d_items/d_labitems label |
| `Event_date` | Measurement timestamp (NULL for non-event variables) |
| `Value` | String representation of the value |
| `Unit` | Unit of measurement — **only** from an explicit unit/uom field in the source (e.g. `valueuom`, `amountuom`, d_items `unitname`). Empty (`''`) when the source has no unit field. **Never** infer from the variable name, the value, or clinical knowledge. |
| `Variable_type` | `timestamp` / `numeric` / `string` / `` (empty) |
| `Is_categorical` | `'1'` if categorical, `'0'` otherwise |
| `Recorder` | (empty in MIMIC) |
| `Recorder_position` | (empty in MIMIC) |
| `Recorder_affiliation` | (empty in MIMIC) |
| `Patient_id` | `subject_id` as STRING |
| `Admission_id` | `hadm_id` as STRING (empty for tables without hadm_id) |
| `Ground_truth` | (empty) |
| `Mapping_info_1` | Primary mapping category (see Mapping Rules) |
| `Mapping_info_2` | Secondary mapping subcategory |

## Mapping Rules (가이드라인 §1.5.2)

| Rule | Condition | Mapping_info_1 | Mapping_info_2 |
|------|-----------|----------------|----------------|
| 3 | Medical codes (ICD, NDC, GSN, HCPCS, DRG) | `medical_code` | NULL |
| 4 | Lab events | `event` | `lab_event` |
| 4 | Chart/vital events | `event` | `chart_event` |
| 4 | I&O events (input/output/ingredient) | `event` | NULL |
| 5 | Date/time columns | `date` | NULL |
| 6 | Diagnosis names (icd_title, chiefcomplaint) | `diagnosis` | NULL |
| 7 | Drug names (medication, drug, name) | `prescription` | `drug` |
| 7 | Prescription info (dose, route, frequency) | `prescription` | `prescription_info` |
| 8 | Procedure events (procedureevents) | `procedure` | NULL |
| — | Clinical notes (discharge summaries) | `note_clinical` | `DIS` / `ADM` / `EME` / `SUR` / NULL |
| — | Radiology reports | `note_rad` | `CXR` / `AXR` / `SXR` / `CCT` / `ACT` / `BCT` / `SCT` / `ECH` / NULL |
| — | All others | NULL | NULL |

## Conversion Strategy by Table Type

### ① Wide Tables (UNPIVOT)
Single row → multiple QUIQ rows (one per column). `Event_date = NULL`.
- **ADMISSIONS**: admittime/dischtime/deathtime → `date`
- **PATIENTS**: dod → `date`
- **TRANSFERS**: intime/outtime → `date`
- **EMAR**: medication → `prescription/drug`
- **PHARMACY**: medication → `prescription/drug`; route/frequency/dose → `prescription/prescription_info`
- **PRESCRIPTIONS**: drug → `prescription/drug`; gsn/ndc/formulary_drug_cd → `medical_code`
- **DIAGNOSES_ICD**: icd_code → `medical_code`
- **HCPCSEVENTS**: hcpcs_cd → `medical_code`; short_description → `procedure`

### ④ Note Tables (MIMIC_IV_NOTE, base + all CTE pattern)
One original note row → 4 QUIQ rows sharing the same Primary_key.

| Table | Variable_name | Event_date | Mapping_info_1 | Variable_ID |
|-------|--------------|------------|----------------|-------------|
| DISCHARGE | `text` | charttime | `note_clinical` (note_type=`DS`) | '' |
| DISCHARGE | `storetime` | NULL | `date` | '' |
| DISCHARGE | `note_type` | NULL | NULL | '' |
| DISCHARGE | `note_seq` | NULL | NULL | '' |
| RADIOLOGY | `text` | charttime | `note_rad` (note_type=`RR`/`AR`) | '' |
| RADIOLOGY | `storetime` | NULL | `date` | '' |
| RADIOLOGY | `note_type` | NULL | NULL | '' |
| RADIOLOGY | `note_seq` | NULL | NULL | '' |
| DISCHARGE_DETAIL | field_name | NULL | `note_clinical` | '' |
| RADIOLOGY_DETAIL | field_name | NULL | `note_rad` | '' |

`discharge_detail` / `radiology_detail` are JOIN-ed to their parent table on `note_id` to obtain `hadm_id`.

### ② Event Tables (base + all CTE pattern)
One original row → multiple QUIQ rows sharing the same Primary_key.

```
q_XXX_base: JOIN d_items/d_labitems → build all columns
q_XXX_all:  UNION ALL with _rtype ordering
  ① main value  → _m1, _m2 (lab/chart/event/procedure)
  ② date cols   → _m1='date', _m2=NULL
  ③ extra cols  → NULL, NULL
```

| Table | Main value column | _m1 | _m2 |
|-------|-------------------|-----|-----|
| LABEVENTS | value/valuenum | `event` | `lab_event` |
| CHARTEVENTS | value/valuenum | `event` | `chart_event` |
| DATETIMEEVENTS | value | `event` | `chart_event` |
| INGREDIENTEVENTS | amount | `event` | NULL |
| INPUTEVENTS | amount | `event` | NULL |
| OUTPUTEVENTS | value | `event` | NULL |
| PROCEDUREEVENTS | value | `procedure` | NULL |

### Event_date Assignment
- `Event_date = charttime or starttime` for the main measurement row
- `Event_date = NULL` for date columns written as rows (storetime, endtime, etc.)
- `Event_date = _ev` for secondary measurements tied to the same time (valuenum, warning, flag)

## BigQuery Implementation

The complete BigQuery SQL is in `scripts/bigquery.sql`. Key BigQuery differences from DuckDB:

| Feature | DuckDB (v5) | BigQuery |
|---------|-------------|---------|
| Functions | `CREATE MACRO` | `CREATE TEMP FUNCTION` |
| Type detection | `TRY_STRPTIME`, `TRY_CAST AS DOUBLE` | `SAFE.PARSE_TIMESTAMP`, `SAFE_CAST AS FLOAT64` |
| String type | `VARCHAR` | `STRING` |
| UNPIVOT | `INTO NAME k VALUE v` | `FOR k IN (cols)` |
| Reserved words | `"value"`, `"name"` | `` `value` ``, `` `name` `` |
| Table names | `hosp.admissions` | `mimiciv_hosp.admissions` |
| Output | `COPY (...) TO 'file.csv'` | Plain `SELECT` |

### How to run with M4 MCP

```python
import os
from google.cloud import bigquery

# NOTE: m4.execute_query blocks multi-statement SQL (SecurityError).
# This SQL uses CREATE TEMP FUNCTION → must use google.cloud.bigquery directly.
skill_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(skill_dir, "scripts/bigquery.sql")) as f:
    sql = f.read()

# parameters.bq_project: BigQuery project ID where MIMIC-IV data is stored
# Change to your institution's project ID if you host your own copy of MIMIC-IV
bq_project = "physionet-data"  # e.g. "cmi-lab" for CMI lab
bq_billing_project = bq_project + "-492906"  # billing project (may differ)
sql = sql.replace("{bq_project}", bq_project)

# Institution-specific dataset name overrides (if needed)
# e.g. cmi-lab uses MIMIC_IV_ED instead of mimiciv_ed
# sql = sql.replace("mimiciv_ed.", "MIMIC_IV_ED.")

# Optional: filter by patient cohort (subject_id list)
# Wraps each source table in a subquery — avoids full table scans
# PIDS = "(10000032, 10000068, 10001217)"  # comma-separated subject_ids
# wide_tables = [
#     "mimiciv_3_1_hosp.admissions", "mimiciv_3_1_hosp.patients",
#     "mimiciv_3_1_hosp.transfers", "mimiciv_3_1_hosp.omr",
#     "mimiciv_3_1_hosp.drgcodes", "mimiciv_3_1_hosp.emar",
#     "mimiciv_3_1_hosp.emar_detail", "mimiciv_3_1_hosp.hcpcsevents",
#     "mimiciv_3_1_hosp.microbiologyevents", "mimiciv_3_1_hosp.pharmacy",
#     "mimiciv_3_1_hosp.poe", "mimiciv_3_1_hosp.poe_detail",
#     "mimiciv_3_1_hosp.prescriptions", "mimiciv_3_1_hosp.services",
#     "mimiciv_3_1_hosp.diagnoses_icd", "mimiciv_3_1_hosp.procedures_icd",
#     "mimiciv_3_1_icu.icustays",
#     "MIMIC_IV_ED.diagnosis", "MIMIC_IV_ED.edstays",
#     "MIMIC_IV_ED.medrecon", "MIMIC_IV_ED.pyxis",
#     "MIMIC_IV_ED.triage", "MIMIC_IV_ED.vitalsign",
# ]
# for tbl in wide_tables:
#     sql = sql.replace(
#         f"    FROM `{bq_project}`.{tbl}\n",
#         f"    FROM (SELECT * FROM `{bq_project}`.{tbl} WHERE subject_id IN {PIDS})\n"
#     )
# for tbl, alias in [
#     ("mimiciv_3_1_hosp.labevents", "le"),
#     ("mimiciv_3_1_icu.chartevents", "ce"),
#     ("mimiciv_3_1_icu.datetimeevents", "de"),
#     ("mimiciv_3_1_icu.ingredientevents", "ie"),
#     ("mimiciv_3_1_icu.inputevents", "ie"),
#     ("mimiciv_3_1_icu.outputevents", "oe"),
#     ("mimiciv_3_1_icu.procedureevents", "pe"),
# ]:
#     sql = sql.replace(
#         f"    FROM `{bq_project}`.{tbl} {alias}\n",
#         f"    FROM (SELECT * FROM `{bq_project}`.{tbl} WHERE subject_id IN {PIDS}) {alias}\n"
#     )

client = bigquery.Client(project=bq_billing_project)
df = client.query(sql).to_dataframe()
# Returns pd.DataFrame with all QUIQ columns

# Save to CSV
output_path = "quiq_output.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved {len(df):,} rows → {output_path}")
```

> **Note**: The SQL uses `CREATE TEMP FUNCTION` (multi-statement). `m4.execute_query()` blocks multi-statement SQL with SecurityError — use `google.cloud.bigquery.Client` directly instead.

## Critical Implementation Notes

1. **Primary_key is NOT globally unique** — it is unique only within each source table's CTE. The combination of `(Original_table_name, Primary_key)` uniquely identifies a source record.

2. **Event_date is selective** — Only clinically meaningful timestamps appear as `Event_date`. Administrative dates (admittime, storetime as standalone rows) use `Event_date = NULL` per §1.5.5.

3. **Empty string vs NULL** — `Value = ''` means the original field was NULL. Used as a filter: `WHERE raw_val IS NOT NULL` removes these in UNPIVOT-based tables.

4. **d_items JOIN** — ICU event tables use `COALESCE(di.label, CAST(itemid AS STRING))` so the Variable_name always has a human-readable label even if d_items has no matching row.

5. **_var_type detection order** — timestamp is checked before numeric. A value like `"2150-05-01 00:00:00"` is classified as `timestamp`, not `numeric`.

6. **Unit comes only from the data, never from inference** — Populate `Unit` solely from an explicit unit-of-measurement field in the source (MIMIC `valueuom` / `amountuom` / `doseuom`, or a d_items `unitname`). If the source record has no unit field, `Unit` MUST be empty (`''`). Do NOT derive a unit from the variable name, the value, or domain knowledge — e.g. do not tag an `LDL` column as `mg/dL` or an `SBP` column as `mmHg` just because those are the usual units. Units are a property of the source data, not of the concept.

7. **Primary_key ordering is numeric** — `Primary_key` is an integer record index (0-based within each source table). When ordering the output, sort by the integer value, not its string form; a lexicographic sort yields `0,1,10,100,2,…` and scrambles records. Within one record keep the variables in the source column order.

8. **Non-MIMIC / generic wide tables** — The same QUIQ schema and Mapping Rules apply to any wide source table, not only MIMIC. Treat it as §① Wide Tables: UNPIVOT each column into one QUIQ row sharing the record's `Primary_key`, map each column via the Mapping Rules table, put the source row's subject/entity id into `Patient_id`, and leave `Variable_ID`, `Recorder*`, `Ground_truth`, and `Unit` (when the source has no unit field) empty. Columns that match no explicit rule fall through to "All others" → `Mapping_info_1/2` empty.

9. **Flat clinical tables → every clinical column is an `event`** — When the source is a flat per-patient clinical table (one row per patient/observation whose columns are all clinical variables — e.g. a cohort or simulation table of labs + vitals + anthropometrics + demographics + condition/outcome flags), do NOT send demographics and binary flags to "All others". Instead map **every** clinical column to an event:
   - Laboratory / blood-serum assays (HDL, LDL, TG, HbA1c, creatinine, urate, CRP, glucose, …) → `event` / `lab_event`
   - Every other clinical measurement — vital signs (SBP, DBP), anthropometrics (BMI, waist-hip ratio), demographics (age, sex), and binary condition/outcome flags (HTN, DM, CKD, Afib, dyslipidemia, family history, smoking, prevalent/incident disease, follow-up time, …) → `event` / `chart_event`
   - Free-text clinical narrative columns → `note_clinical`
   - Only true identifier/administrative columns stay out of the value rows (subject id → `Patient_id`).

   Binary/categorical flags are still events: keep `Variable_type = 'numeric'` (or `string`) and `Is_categorical = '1'`. This convention is specific to flat clinical tables; it does NOT change the general MIMIC Mapping Rules above (in MIMIC, demographics from `patients` remain "All others").

## References

- LYDUS 품질관리 프로그램 활용 가이드라인 제4권 §1.5~§1.6 (QUIQ table specification and Mapping rules)
- Johnson AEW et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10:1.