# LYDUS — Healthcare Data Quality Assessment Code

This repository is part of the [LYDUS project](https://lydus.org/). It provides Python scripts, configuration files, and Agent Skills for healthcare data quality assessment and data preparation.

> **Find files without opening each folder:** Use the linked tables below to access Python scripts, YAML configurations, notebooks, and Agent Skills directly from this page.

## Contents

- [Repository overview](#repository-overview)
- [Single-center quality metrics](#single-center-quality-metrics)
- [Multi-center quality metrics](#multi-center-quality-metrics)
- [How to run](#how-to-run)
- [Agent Skills](#agent-skills)
- [QUIQ conversion examples](#quiq-conversion-examples)
- [Notes](#notes)

## Repository overview

| Folder | Purpose | Main contents |
| --- | --- | --- |
| [LYDUS_Single_Center](LYDUS_Single_Center/) | Quality assessment within a single healthcare dataset | Python metrics, YAML configurations, Agent Skills, and LLaMA 3 variants |
| [LYDUS_Multi_Center](LYDUS_Multi_Center/) | Comparison of healthcare data across centers | Structured metrics 01–07, unstructured metrics B-1/B-2, and three Agent Skills |
| [LYDUS_codes_to_QUIQ](LYDUS_codes_to_QUIQ/) | Examples of converting source data into QUIQ format and related agent guidance | Jupyter notebooks, QUIQ/VIA Agent Skills and references |

**File types:** `*.py` = executable Python code · `*.yml` = configuration · `SKILL.md` = agent instructions · `PROVENANCE.yaml` = skill provenance metadata · `*.ipynb` = Jupyter notebook.

## Single-center quality metrics

Each row links to the available Python implementation and YAML configuration. The **LLaMA 3** column is populated only for the seven metrics with a separate local-model version. For Agent Skill links, see the consolidated [Agent Skills](#agent-skills) section. `—` means that no corresponding file is included in this repository snapshot.

| Metric | Python | YAML config | LLaMA 3 Python |
| --- | --- | --- | --- |
| Completeness | [`.py`](<LYDUS_Single_Center/py/LYDUS_Completeness.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_completeness.yml>) | — |
| Classification | [`.py`](<LYDUS_Single_Center/py/LYDUS_Classification.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_classification.yml>) | — |
| Fidelity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Fidelity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_fidelity.yml>) | — |
| Preciseness | [`.py`](<LYDUS_Single_Center/py/LYDUS_Preciseness.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_preciseness.yml>) | — |
| Date Validity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Date_Validity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_date_validity.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Date_Validity_LLaMA3.py>) |
| Format Validity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Format_Validity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_format_validity.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Format_Validity_LLaMA3.py>) |
| Range Validity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Range_Validity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_range_validity.yml>) | — |
| Sequence Validity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Sequence_Validity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_sequence_validity.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Sequence_Validity_LLaMA3.py>) |
| Cross-sectional Consistency | [`.py`](<LYDUS_Single_Center/py/LYDUS_Cross_Sectional_Consistency.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_cross_sectional_consistency.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Cross_Sectional_Consistency_LLaMA3.py>) |
| Time-series Consistency | [`.py`](<LYDUS_Single_Center/py/LYDUS_Time_Series_Consistency.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_time_series_consistency.yml>) | — |
| Logical Accuracy | [`.py`](<LYDUS_Single_Center/py/LYDUS_Logical_Accuracy.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_logical_accuracy.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Logical_Accuracy_LLaMA3.py>) |
| Note Accuracy | [`.py`](<LYDUS_Single_Center/py/LYDUS_Note_Accuracy.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_note_accuracy.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Note_Accuracy_LLaMA3.py>) |
| Note Fidelity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Note_Fidelity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_note_fidelity.yml>) | [`.py`](<LYDUS_Single_Center/py%20(LLaMA3%20ver)/LYDUS_Note_Fidelity_LLaMA3.py>) |
| Class Diversity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Class_Diversity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_class_diversity.yml>) | — |
| Instance Diversity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Instance_Diversity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_instance_diversity.yml>) | — |
| Sentence Diversity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Sentence_Diversity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_sentence_diversity.yml>) | — |
| Vocabulary Diversity | [`.py`](<LYDUS_Single_Center/py/LYDUS_Vocabulary_Diversity.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_vocabulary_diversity.yml>) | — |
| Bias Detection | [`.py`](<LYDUS_Single_Center/py/LYDUS_Bias_Detection.py>) | [`.yml`](<LYDUS_Single_Center/yml/config_bias_detection.yml>) | — |

## Multi-center quality metrics

### Structured data

The structured script implements multiple comparison metrics in one Python file. Scripts are stored under `LYDUS_Multi_Center/py/`, and configurations under `LYDUS_Multi_Center/yml/`.

| Analysis | Python script | YAML configuration | What it evaluates |
| --- | --- | --- | --- |
| Structured 01–07 | [`LYDUS_Multi_Structured.py`](<LYDUS_Multi_Center/py/LYDUS_Multi_Structured.py>) | [`config_multi_structured.yml`](<LYDUS_Multi_Center/yml/config_multi_structured.yml>) | Cross-center quality comparisons for structured data |

The structured script covers:

| ID | Quality metric |
| --- | --- |
| 01 | Variable name consistency |
| 02 | Categorical variable distribution homogeneity |
| 03 | Categorical variable distribution similarity (Jensen–Shannon divergence) |
| 04 | Continuous variable distribution homogeneity |
| 05 | Continuous variable distribution shape consistency |
| 06 | Measurement unit consistency |
| 07 | Medical code consistency |

### Unstructured data

| ID | Python script | YAML configuration | What it evaluates |
| --- | --- | --- | --- |
| B-1 | [`LYDUS_Multi_Unstructured_1.py`](<LYDUS_Multi_Center/py/LYDUS_Multi_Unstructured_1.py>) | [`config_multi_unstructured_1.yml`](<LYDUS_Multi_Center/yml/config_multi_unstructured_1.yml>) | Note style distinguishability between centers (TF–IDF and classification). |
| B-2 | [`LYDUS_Multi_Unstructured_2.py`](<LYDUS_Multi_Center/py/LYDUS_Multi_Unstructured_2.py>) | [`config_multi_unstructured_2.yml`](<LYDUS_Multi_Center/yml/config_multi_unstructured_2.yml>) | Differences in note-length distributions (Wasserstein distance and normalized WD). |

## How to run

1. Open the corresponding **Python** and **YAML** links in the tables above.
2. Download the required files, prepare input data, and edit the paths and options in the YAML configuration.
3. Install the Python dependencies imported by the chosen script in your local Python environment.
4. Run the script with its matching configuration file:

```bash
python LYDUS_Metric_Name.py --config config_metric_name.yml
```

## Agent Skills

Agent Skills provide task-specific instructions and supporting resources for AI agents working with LYDUS. They are organized into three groups: **QUIQ/VIA (2 skills)**, **Single-center (18 skills)**, and **Multi-center (3 skills)**. They guide data preparation and quality assessment workflows; the associated scripts and execution environments are still needed for code-based analyses.

### Agent Skill overview

| Category | Scope | Skill directory | Availability |
| --- | --- | --- | --- |
| QUIQ / VIA | Data preparation and representation guidance | [`LYDUS_codes_to_QUIQ/skill/`](LYDUS_codes_to_QUIQ/skill/) | Available |
| Single-center | Metric-specific quality assessment instructions | [`LYDUS_Single_Center/skill/`](LYDUS_Single_Center/skill/) | 18 skills available |
| Multi-center | Cross-center structured and unstructured data assessment | [`LYDUS_Multi_Center/skill/`](LYDUS_Multi_Center/skill/) | 3 skills available |

### QUIQ / VIA

The QUIQ skill describes conversion of MIMIC-IV data into the QUIQ long-format table. The VIA skill generates variable descriptions using a QUIQ dataset and **user-provided documentation** (e.g., a data dictionary or schema document). The VIA skill requires that documentation as input.

| Skill | Instructions | Supporting files |
| --- | --- | --- |
| QUIQ | [`SKILL.md`](LYDUS_codes_to_QUIQ/skill/quiq/SKILL.md) | [`PROVENANCE.yaml`](LYDUS_codes_to_QUIQ/skill/quiq/PROVENANCE.yaml) |
| VIA | [`SKILL.md`](LYDUS_codes_to_QUIQ/skill/via/SKILL.md) | [`PROVENANCE.yaml`](LYDUS_codes_to_QUIQ/skill/via/PROVENANCE.yaml) 
### Single-center

The following 18 Agent Skills correspond to the metrics in [Single-center quality metrics](#single-center-quality-metrics). Each Skill directory includes `SKILL.md` and `PROVENANCE.yaml`. Requirements vary by metric; consult the linked instructions for data inputs, dependencies, and model/API requirements.

| Metric | Skill instructions | Metric | Skill instructions |
| --- | --- | --- | --- |
| Completeness | [`SKILL.md`](<LYDUS_Single_Center/skill/completeness/SKILL.md>) | Classification | [`SKILL.md`](<LYDUS_Single_Center/skill/classification/SKILL.md>) |
| Fidelity | [`SKILL.md`](<LYDUS_Single_Center/skill/fidelity/SKILL.md>) | Preciseness | [`SKILL.md`](<LYDUS_Single_Center/skill/preciseness/SKILL.md>) |
| Date Validity | [`SKILL.md`](<LYDUS_Single_Center/skill/date-validity/SKILL.md>) | Format Validity | [`SKILL.md`](<LYDUS_Single_Center/skill/format-validity/SKILL.md>) |
| Range Validity | [`SKILL.md`](<LYDUS_Single_Center/skill/range-validity/SKILL.md>) | Sequence Validity | [`SKILL.md`](<LYDUS_Single_Center/skill/sequence-validity/SKILL.md>) |
| Cross-sectional Consistency | [`SKILL.md`](<LYDUS_Single_Center/skill/cross-sectional-consistency/SKILL.md>) | Time-series Consistency | [`SKILL.md`](<LYDUS_Single_Center/skill/time-series-consistency/SKILL.md>) |
| Logical Accuracy | [`SKILL.md`](<LYDUS_Single_Center/skill/logical-accuracy/SKILL.md>) | Note Accuracy | [`SKILL.md`](<LYDUS_Single_Center/skill/note-accuracy/SKILL.md>) |
| Note Fidelity | [`SKILL.md`](<LYDUS_Single_Center/skill/note-fidelity/SKILL.md>) | Class Diversity | [`SKILL.md`](<LYDUS_Single_Center/skill/class-diversity/SKILL.md>) |
| Instance Diversity | [`SKILL.md`](<LYDUS_Single_Center/skill/instance-diversity/SKILL.md>) | Sentence Diversity | [`SKILL.md`](<LYDUS_Single_Center/skill/sentence-diversity/SKILL.md>) |
| Vocabulary Diversity | [`SKILL.md`](<LYDUS_Single_Center/skill/vocabulary-diversity/SKILL.md>) | Bias Detection | [`SKILL.md`](<LYDUS_Single_Center/skill/bias-detection/SKILL.md>) |

### Multi-center

The following three Agent Skills support cross-center quality assessment of structured and unstructured healthcare data. The corresponding Python scripts and YAML configurations are linked under [Multi-center quality metrics](#multi-center-quality-metrics).

| Metric | Skill Instructions |
| --- | --- |
| Structured data (01–07) | [`SKILL.md`](LYDUS_Multi_Center/skill/multi-structured/SKILL.md) |
| Unstructured data (B-1) | [`SKILL.md`](LYDUS_Multi_Center/skill/multi-unstructured-1/SKILL.md) |
| Unstructured data (B-2) | [`SKILL.md`](LYDUS_Multi_Center/skill/multi-unstructured-2/SKILL.md) |

**Note:** These skills describe workflows and required inputs. Review configuration settings and the current scripts before execution.

## QUIQ conversion examples

| Example notebook | Purpose |
| --- | --- |
| [`MIMIC_to_QUIQ.ipynb`](<LYDUS_codes_to_QUIQ/MIMIC_to_QUIQ.ipynb>) | Example conversion from MIMIC source data to QUIQ. |
| [`SYNTHEA_to_QUIQ.ipynb`](<LYDUS_codes_to_QUIQ/SYNTHEA_to_QUIQ.ipynb>) | Example conversion from Synthea source data to QUIQ. |

## Notes

- The repository is updated over time. Confirm the latest versions of scripts and configurations before running analyses.
