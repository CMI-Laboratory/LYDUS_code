# MIMIC-IV Data Dictionary (Bundled Default)

This is the **bundled DEFAULT MIMIC-IV data dictionary** shipped with the VIA skill.
It covers the commonly-used columns of the MIMIC-IV `hosp` and `icu` modules (v2.2 / v3.1).
Users may upload their own documentation file to **override or extend** these defaults.

## Identifier Hierarchy

MIMIC-IV links records through a small set of shared keys:

- **subject_id** — unique identifier for an individual patient. Stable across all admissions and stays.
- **hadm_id** — unique identifier for a single hospital admission (one patient may have many).
- **stay_id** — unique identifier for a single ICU stay (nested within a hospital admission).
- **itemid** — identifier for a measurement/observation concept; resolved to a human-readable label via `d_items` (ICU) or `d_labitems` (labs).

---

# hosp module

## patients

| Column | Description |
| --- | --- |
| subject_id | Unique identifier for an individual patient across the whole database. |
| gender | Patient's administrative/genotypical sex (M or F). |
| anchor_age | Patient's age in the `anchor_year`. A value of 0 indicates the patient was under 1 year old; ages over 89 are aggregated to protect identity. |
| anchor_year | A deidentified (shifted) year used as the reference point for the patient's timeline; real dates are offset consistently per patient. |
| anchor_year_group | A 3-year range (e.g. "2011 - 2013") indicating the real-world period into which `anchor_year` maps. |
| dod | Date of death, sourced from hospital records and state death registries; null if the patient is not known to have died. |

## admissions

| Column | Description |
| --- | --- |
| subject_id | Patient identifier linking the admission to `patients`. |
| hadm_id | Unique identifier for this hospital admission. |
| admittime | Date and time the patient was admitted to the hospital. |
| dischtime | Date and time the patient was discharged from the hospital. |
| deathtime | Date and time of in-hospital death, if it occurred during this admission; otherwise null. |
| admission_type | Category/urgency of the admission (e.g. ELECTIVE, URGENT, EW EMER., OBSERVATION ADMIT). |
| admission_location | Where the patient came from prior to admission (e.g. EMERGENCY ROOM, PHYSICIAN REFERRAL, TRANSFER FROM HOSPITAL). |
| discharge_location | Where the patient was discharged to (e.g. HOME, SNF, REHAB, DIED). |
| insurance | Payer/insurance category for the admission (e.g. Medicare, Medicaid, Other). |
| language | Patient's primary/preferred language as recorded. |
| marital_status | Patient's marital status (e.g. SINGLE, MARRIED, WIDOWED, DIVORCED). |
| race | Patient's self-reported race/ethnicity. |
| edregtime | Date and time the patient was registered in the emergency department, if applicable. |
| edouttime | Date and time the patient left the emergency department, if applicable. |
| hospital_expire_flag | Binary flag: 1 if the patient died during this hospital admission, 0 otherwise. |

## transfers

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission identifier; may be null for events (e.g. ED) not tied to an inpatient admission. |
| transfer_id | Unique identifier for a physical location/care transfer event. |
| eventtype | Type of transfer event (admit, transfer, or discharge). |
| careunit | The care unit the patient was located in during this transfer (e.g. MICU, SICU, Med/Surg ward). |
| intime | Date and time the patient entered this care unit/location. |
| outtime | Date and time the patient left this care unit/location. |

## diagnoses_icd

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission the diagnosis is billed under. |
| seq_num | Priority/order of the diagnosis for the admission; lower numbers are generally more important (e.g. principal diagnosis). |
| icd_code | The ICD diagnosis code assigned (billed) for the admission. |
| icd_version | ICD coding system version for the code, either 9 or 10. |

## procedures_icd

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission the procedure is billed under. |
| seq_num | Priority/order of the procedure within the admission. |
| chartdate | Date the procedure was performed/charted. |
| icd_code | The ICD procedure code assigned for the admission. |
| icd_version | ICD coding system version for the code, either 9 or 10. |

## prescriptions

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission during which the medication was prescribed. |
| pharmacy_id | Identifier linking to the corresponding `pharmacy` record for this order. |
| starttime | Date and time the prescription/medication was scheduled to start. |
| stoptime | Date and time the prescription/medication was scheduled to stop. |
| drug | Free-text name of the prescribed medication. |
| formulary_drug_cd | Hospital formulary code for the drug. |
| gsn | Generic Sequence Number, a coded identifier for the drug. |
| ndc | National Drug Code identifying the specific product/packaging. |
| prod_strength | Free-text description of the product strength/form (e.g. "500mg Tablet"). |
| dose_val_rx | The prescribed dose amount (value). |
| dose_unit_rx | Unit of measurement for the prescribed dose (e.g. mg, mL). |
| route | Route of administration (e.g. PO, IV, IM, SC). |

## labevents

| Column | Description |
| --- | --- |
| labevent_id | Unique identifier for a single laboratory measurement row. |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission the lab is associated with; may be null if drawn outside an inpatient admission. |
| specimen_id | Identifier grouping measurements drawn from the same specimen/collection. |
| itemid | Identifier for the lab concept, resolved to a label via `d_labitems`. |
| charttime | Date and time the specimen was charted/measured. |
| value | The lab result as originally recorded (may be text or numeric). |
| valuenum | The numeric form of `value` when it can be represented as a number; null for non-numeric results. |
| valueuom | Unit of measurement for the lab value. |
| ref_range_lower | Lower bound of the reference (normal) range for this measurement. |
| ref_range_upper | Upper bound of the reference (normal) range for this measurement. |
| flag | Abnormality indicator; typically "abnormal" when the value is outside the reference range. |

## d_labitems

| Column | Description |
| --- | --- |
| itemid | Identifier for a laboratory concept, referenced by `labevents`. |
| label | Human-readable name of the laboratory measurement (e.g. "Creatinine", "Hemoglobin"). |
| fluid | The specimen/fluid type measured (e.g. Blood, Urine). |
| category | Broad grouping of the lab (e.g. Chemistry, Hematology, Blood Gas). |

## d_icd_diagnoses

| Column | Description |
| --- | --- |
| icd_code | An ICD diagnosis code. |
| icd_version | ICD version for the code, 9 or 10. |
| long_title | Full textual description of the diagnosis code. |

## d_icd_procedures

| Column | Description |
| --- | --- |
| icd_code | An ICD procedure code. |
| icd_version | ICD version for the code, 9 or 10. |
| long_title | Full textual description of the procedure code. |

## microbiologyevents

| Column | Description |
| --- | --- |
| microevent_id | Unique identifier for a microbiology measurement row. |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission associated with the culture, if applicable. |
| chartdate | Date the specimen was charted. |
| charttime | Date and time the specimen was charted, when available. |
| spec_itemid | Identifier for the specimen type tested. |
| spec_type_desc | Textual description of the specimen collected (e.g. BLOOD CULTURE, URINE). |
| org_itemid | Identifier for the organism grown/isolated, if any. |
| org_name | Name of the organism isolated; null if no growth. |
| ab_itemid | Identifier for the antibiotic tested against the organism. |
| ab_name | Name of the antibiotic tested for susceptibility. |
| dilution_text | Measured dilution/MIC value for the antibiotic as text. |
| interpretation | Susceptibility interpretation (S = susceptible, R = resistant, I = intermediate, P = pending). |

---

# icu module

## icustays

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission that contains this ICU stay. |
| stay_id | Unique identifier for this ICU stay. |
| first_careunit | The ICU/care unit the patient was first admitted to for this stay. |
| last_careunit | The ICU/care unit the patient was in at the end of this stay. |
| intime | Date and time the patient was admitted to the ICU. |
| outtime | Date and time the patient was discharged from the ICU. |
| los | Length of stay in the ICU, measured in fractional days. |

## chartevents

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission associated with the charted event. |
| stay_id | ICU stay during which the event was charted. |
| itemid | Identifier for the charted concept, resolved to a label via `d_items`. |
| charttime | Date and time the observation was charted (time of measurement). |
| storetime | Date and time the observation was validated/stored in the system. |
| value | The charted value as originally recorded (text or numeric). |
| valuenum | Numeric form of `value` when applicable; null for non-numeric values. |
| valueuom | Unit of measurement for the value. |
| warning | Flag indicating the value triggered a monitoring/validation warning (e.g. out of expected range). |

## d_items

| Column | Description |
| --- | --- |
| itemid | Identifier for an ICU measurement/observation concept. |
| label | Human-readable name of the concept (e.g. "Heart Rate", "Arterial Blood Pressure systolic"). |
| abbreviation | Common abbreviation for the concept as shown in the clinical system. |
| linksto | Name of the table where data for this item is stored (e.g. chartevents, inputevents). |
| category | Grouping/category of the item (e.g. Routine Vital Signs, Labs, Medications). |
| unitname | Default unit of measurement associated with the item, when applicable. |
| param_type | Data type of the item's value (e.g. Numeric, Text, Date, Checkbox). |

## inputevents

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission associated with the input. |
| stay_id | ICU stay during which the input (e.g. fluid, drug) was administered. |
| itemid | Identifier for the administered substance, resolved via `d_items`. |
| starttime | Date and time the administration started. |
| endtime | Date and time the administration ended. |
| amount | Amount of the substance administered over the interval. |
| amountuom | Unit of measurement for `amount` (e.g. mL, mg). |
| rate | Rate of administration, when the input was given continuously. |
| rateuom | Unit of measurement for `rate` (e.g. mL/hour, mcg/kg/min). |
| ordercategoryname | Category describing the order/administration type (e.g. Continuous IV, Bolus). |

## outputevents

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission associated with the output. |
| stay_id | ICU stay during which the output was recorded. |
| itemid | Identifier for the output type (e.g. urine output), resolved via `d_items`. |
| charttime | Date and time the output was charted. |
| value | Amount of output recorded. |
| valueuom | Unit of measurement for the output value (e.g. mL). |

## procedureevents

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission associated with the procedure. |
| stay_id | ICU stay during which the procedure occurred. |
| itemid | Identifier for the procedure concept, resolved via `d_items`. |
| starttime | Date and time the procedure started. |
| endtime | Date and time the procedure ended. |
| value | Value/quantity associated with the procedure (e.g. duration or count). |
| valueuom | Unit of measurement for `value`. |

## datetimeevents

| Column | Description |
| --- | --- |
| subject_id | Patient identifier. |
| hadm_id | Hospital admission associated with the event. |
| stay_id | ICU stay during which the datetime observation was charted. |
| itemid | Identifier for the concept being recorded as a date/time, resolved via `d_items`. |
| charttime | Date and time the observation was charted. |
| value | The recorded date/time value for the concept (this table stores observations whose values are themselves dates/times). |
| valueuom | Unit of measurement/description for the value. |
