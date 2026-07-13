import pandas as pd
import numpy as np
import openai
import json
import re
import random
import argparse
import yaml
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.stats import shapiro, skew, kurtosis
import numpy as np
from diptest import diptest

### Structured 01 - Variable Name Consistency

def normalize(x):
  return str(x).lower().strip()

def chunk_list(lst, size=50):
  for i in range(0, len(lst), size):
      yield lst[i:i+size]

def extract_variable_lists(df_A, df_B):
  A_list = (
      df_A['Variable_name']
      .dropna()
      .apply(normalize)
      .unique()
      .tolist()
  )

  B_list = (
      df_B['Variable_name']
      .dropna()
      .apply(normalize)
      .unique()
      .tolist()
  )

  return A_list, B_list

def build_pair_mapping_with_llm(client, model, A_list, B_list):

  prompt = f"""
  You are a clinical variable harmonization system.

  Your task is to match variables from Hospital A to variables from Hospital B.

  IMPORTANT RULES:
  1. Only match variables that represent the SAME clinical concept.
  2. Output MUST be valid JSON.
  3. DO NOT include explanations.
  4. Multiple matches are allowed if they represent the same clinical concept.
  5. Be conservative and avoid weak semantic matches.
  6. If no match exists, do not include it.
  7. Be conservative (avoid false matches).

  Output format:
  [
    {{"A": "blood sugar", "B": "glucose", "score": 0.97}},
    {{"A": "heart rate", "B": "hr", "score": 0.88}}
  ]

  Hospital A variables:
  {A_list}

  Hospital B variables:
  {B_list}
  """

  response = client.responses.create(
      model=model,
      input =[{"role": "user", "content": prompt}],
  )

  return response.output_text

def load_pairs(text):
  cleaned = text.strip()
  cleaned = re.sub(r"```.*?\n", "", cleaned)
  cleaned = cleaned.replace("```", "").strip()

  match = re.search(r"\[.*\]", cleaned, re.DOTALL)
  if match:
      cleaned = match.group(0)

  try:
      return json.loads(cleaned)
  except:
      print("Parsing Failure")
      print(cleaned[:500])
      return []

def run_llm_pair_matching(client, model, A_list, B_list, chunk_size=50):

  all_pairs = []

  for A_chunk in chunk_list(A_list, chunk_size):
      response = build_pair_mapping_with_llm(client, model, A_chunk, B_list)
      pairs = load_pairs(response)
      all_pairs.extend(pairs)

  pair_df = pd.DataFrame(all_pairs)

  if pair_df.empty:
      return pair_df

  # 컬럼 정리
  pair_df['A'] = pair_df['A'].apply(normalize)
  pair_df['B'] = pair_df['B'].apply(normalize)
  pair_df['score'] = pd.to_numeric(pair_df['score'], errors = 'coerce')
  
  pair_df = pair_df[pair_df['score'] >= 0.8]
  pair_df = pair_df.sort_values(by = ['score', 'A', 'B'], ascending = [False, True, True]).reset_index(drop = True)

  pair_df = pair_df.drop_duplicates(subset=['A', 'B'], keep='first')
  
  return pair_df

def build_result_dataframe(pair_df):

  df = pair_df.copy()

  df['A_var_name'] = df['A']
  df['B_var_name'] = df['B']

  df['Is_hetero'] = df.apply(
      lambda row: row['A_var_name'] != row['B_var_name'],
      axis=1
  )

  return df[['A_var_name', 'B_var_name', 'Is_hetero']]

def run_full_pipeline(df_A, df_B, client, model):

    A_list, B_list = extract_variable_lists(df_A, df_B)
  
    pair_df = run_llm_pair_matching(client, model, A_list, B_list)

    result_df = build_result_dataframe(pair_df)

    return result_df

### Structured 02 - Categorical Value Consistency

def build_value_clusters_with_llm(client, model, variable_name, value_list):

  SYSTEM_CONTENT = f"""
  Goal: Group values that represent the SAME real-world concept.

  You are given values from ONE variable. All values refer to the same type of concept.

  Variable name: {variable_name}

  Rules:
  - Group values if they represent the same real-world meaning.
  - Do NOT change or normalize the original strings.
  - Keep original strings EXACTLY as given.
  - If meanings are clearly different → separate groups.

  IMPORTANT:
  - Abbreviations, language differences, and formatting variations MUST be grouped if they mean the same thing.
  - Differences in suffixes, language, spacing, or capitalization should NOT prevent grouping.

  Examples you MUST follow:
  - 'M' and 'Male' → SAME
  - 'F' and 'Female' → SAME
  - 'Y' and 'Yes' → SAME
  - 'N' and 'No' → SAME

  CRITICAL GUIDELINES:
  - If two values are very likely to have the same meaning, you MUST group them.
  - It is better to slightly over-group than to under-group.
  - Do NOT split values that only differ by language or formatting.

  🔥 HARD CONSTRAINTS:
  - You MUST include EVERY input value in EXACTLY ONE category.
  - NO value should be missing.
  - NO value should appear in multiple categories.
  - If unsure, create a new category instead of dropping the value.

  Output format (STRICT):
  category1: 'A','A형'
  category2: 'B','B형'
  category3: 'O','O형'
  """

  prompt = f"""
  Input values:
  {value_list}
  """

  response = client.chat.completions.create(
      model=model,
      messages=[
          {"role": "system", "content": SYSTEM_CONTENT},
          {"role": "user", "content": prompt}
      ]
  )

  return response.choices[0].message.content

def parse_cluster_output(text):
  clusters = {}

  lines = text.strip().split("\n")

  for line in lines:
      if ":" not in line:
          continue

      key, values_str = line.split(":", 1)
      key = key.strip().lower()

      values = re.findall(r"'(.*?)'", values_str)

      if len(values) > 0:
          clusters[key] = values

  return clusters

def compute_surface_mismatch_from_pairs(df_A, df_B, pair_df, client, model):

  results = []

  for _, row in pair_df.iterrows(): 
      var_A = row['A_var_name']
      var_B = row['B_var_name']
      variable_name = f'var_A{var_B}'

      VA = df_A[df_A['Variable_name'].str.lower().str.strip() == var_A]['Value'] \
              .dropna().unique().tolist()

      VB = df_B[df_B['Variable_name'].str.lower().str.strip() == var_B]['Value'] \
              .dropna().unique().tolist()

      union_values = list(set(VA + VB))

      cluster_text = build_value_clusters_with_llm(client, model, variable_name, union_values)
      cluster_dict = parse_cluster_output(cluster_text)

      mismatch_count = 0
      total_clusters = 0

      cluster_details = []

      for c, values in cluster_dict.items():

          A_surface = set([v for v in values if v in VA])
          B_surface = set([v for v in values if v in VB])

          if len(A_surface) > 0 or len(B_surface) > 0:

              total_clusters += 1

              is_mismatch = A_surface != B_surface

              if is_mismatch:
                  mismatch_count += 1

              cluster_details.append({
                  "cluster": c,
                  "A_values": list(A_surface),
                  "B_values": list(B_surface),
                  "is_mismatch": is_mismatch
              })

      surface_match_rate = (
          1 - (mismatch_count / total_clusters)
          if total_clusters > 0 else 0
      )

      results.append({
          "A_var_name": var_A,
          "B_var_name": var_B,
          "surface_match_rate": surface_match_rate * 100,
          "n_clusters": total_clusters,
          "n_mismatch": mismatch_count,
          "cluster_detail": cluster_details
      })

  return pd.DataFrame(results)

### Structured 03 - Categorical Variable Distribution Homogeneity

def js_divergence(p, q):
  p = np.array(p)
  q = np.array(q)

  # zero smoothing
  p = p + 1e-8
  q = q + 1e-8

  p = p / p.sum()
  q = q / q.sum()

  m = 0.5 * (p + q)

  def kl(a, b):
      return np.sum(a * np.log(a / b))

  return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def get_site_representative(values, site_values):

  candidates = [v for v in values if v in site_values]

  if not candidates:
      return None

  return max(set(candidates), key=candidates.count)


def build_value_to_cluster_from_surface(cluster_detail):

  value_to_cluster = {}

  for item in cluster_detail:
      c = item["cluster"]

      for v in item["A_values"] + item["B_values"]:
          if v not in value_to_cluster:
              value_to_cluster[v] = c

  return value_to_cluster

def build_cluster_labels(cluster_dict, VA, VB):

  A_labels = {}
  B_labels = {}

  for c, values in cluster_dict.items():
      A_labels[c] = get_site_representative(values, VA)
      B_labels[c] = get_site_representative(values, VB)

  return A_labels, B_labels

def compute_categorical_distribution_heterogeneity(df_A, df_B, df_surface):

  results = []

  for _, row in df_surface.iterrows():

      var_A = row['A_var_name']
      var_B = row['B_var_name']
      cluster_detail = row['cluster_detail']

      VA = df_A[
          df_A['Variable_name'].str.lower().str.strip() == var_A
      ]['Value'].dropna().tolist()

      VB = df_B[
          df_B['Variable_name'].str.lower().str.strip() == var_B
      ]['Value'].dropna().tolist()

      if len(VA) == 0 or len(VB) == 0:
          continue

      value_to_cluster = build_value_to_cluster_from_surface(cluster_detail)

      A_norm = [value_to_cluster[v] for v in VA if v in value_to_cluster]
      B_norm = [value_to_cluster[v] for v in VB if v in value_to_cluster]

      if len(A_norm) == 0 or len(B_norm) == 0:
          continue

      common_categories = list(set(A_norm) & set(B_norm))

      if len(common_categories) < 2:
          jsd = np.nan
          A_dist, B_dist = {}, {}

      else:
          def get_distribution(values, categories):
            counts = np.array([values.count(c) for c in categories])
            return counts / counts.sum()

          P = get_distribution(A_norm, common_categories)
          Q = get_distribution(B_norm, common_categories)

          jsd = js_divergence(P, Q)

          def get_representative(cluster, values):
              return max(
                  [v for v in values if value_to_cluster.get(v) == cluster],
                  key=lambda x: values.count(x),
                  default=None
              )

          A_dist, B_dist = {}, {}

          for c, p, q in zip(common_categories, P, Q):

              A_label = get_representative(c, VA)
              B_label = get_representative(c, VB)

              if A_label:
                  A_dist[A_label] = p

              if B_label:
                  B_dist[B_label] = q

      results.append({
          "A_var_name": var_A,
          "B_var_name": var_B,
          "A_dist": A_dist,
          "B_dist": B_dist,
          "js_divergence": jsd,
      })

  return pd.DataFrame(results)

### Structured 04 - Continuous Variable Distribution Homogeneity

def clean_value(x):
    try:
        return float(x)
    except:
        return np.nan

def build_pair_distribution(df_A, df_B, pair_df):

  results = []

  for _, row in pair_df.iterrows():

      var_A = row['A_var_name']
      var_B = row['B_var_name']

      A_vals = df_A.loc[
          (df_A['var_norm'] == var_A) &
          (df_A['value_clean'].notna()),
          'value_clean'
      ].tolist()

      B_vals = df_B.loc[
          (df_B['var_norm'] == var_B) &
          (df_B['value_clean'].notna()),
          'value_clean'
      ].tolist()

      results.append({
          'A_var_name': var_A,
          'B_var_name': var_B,
          'A_values': A_vals,
          'B_values': B_vals
      })

  return pd.DataFrame(results)

def compute_wd(a, b):
  if isinstance(a, list) and isinstance(b, list) and len(a) > 0 and len(b) > 0:
      return wasserstein_distance(a, b)
  return np.nan

def normalize_wd(row):
  vals = []

  if isinstance(row['A_values'], list):
      vals += row['A_values']
  if isinstance(row['B_values'], list):
      vals += row['B_values']

  if len(vals) == 0:
      return np.nan

  std = np.std(vals)
  if std == 0:
      return 0

  return row['wasserstein_dist'] / std

def compute_continuous_variable_distribution_homogeniety(df_A, df_B, pair_df):
  quiq_a_ = df_A.copy()
  quiq_b_ = df_B.copy()

  quiq_a_['var_norm'] = quiq_a_['Variable_name'].apply(normalize)
  quiq_b_['var_norm'] = quiq_b_['Variable_name'].apply(normalize)
  quiq_a_['value_clean'] = quiq_a_['Value'].apply(clean_value)
  quiq_b_['value_clean'] = quiq_b_['Value'].apply(clean_value)

  merged_dist = build_pair_distribution(quiq_a_, quiq_b_, pair_df)

  merged_dist['wasserstein_dist'] = merged_dist.apply(
      lambda row: compute_wd(row['A_values'], row['B_values']),
      axis=1
  )

  merged_dist['wd_normalized'] = merged_dist.apply(normalize_wd, axis=1)

  merged_dist_ = merged_dist.drop(['A_values', 'B_values'], axis=1)

  merged_dist_ = merged_dist_[[
      'A_var_name',
      'B_var_name',
      'wasserstein_dist',
      'wd_normalized',
  ]]

  return merged_dist_

### Structured 05 - Continuous Variable Distribution Shape Consistency

def extract_distribution_features(values):

    if not isinstance(values, list) or len(values) < 20:
        return {
            "skew": np.nan,
            "kurtosis": np.nan,
            "dip": np.nan
        }

    values = np.array(values)

    # skewness
    sk = skew(values)

    # kurtosis
    kt = kurtosis(values)

    # dip statistic
    try:
        dip, dip_p = diptest(values)
    except:
        dip = np.nan

    return {
        "skew": sk,
        "kurtosis": kt,
        "dip": dip
    }

def compute_continuous_variable_distribution_shape_consistency(df_A, df_B, pair_df):
  quiq_a_ = df_A.copy()
  quiq_b_ = df_B.copy()

  quiq_a_['var_norm'] = quiq_a_['Variable_name'].apply(normalize)
  quiq_b_['var_norm'] = quiq_b_['Variable_name'].apply(normalize)
  quiq_a_['value_clean'] = quiq_a_['Value'].apply(clean_value)
  quiq_b_['value_clean'] = quiq_b_['Value'].apply(clean_value)

  merged_dist = build_pair_distribution(quiq_a_, quiq_b_, pair_df)

  merged_dist['A_feat'] = merged_dist['A_values'].apply(extract_distribution_features)
  merged_dist['B_feat'] = merged_dist['B_values'].apply(extract_distribution_features)

  merged_dist['A_skew'] = merged_dist['A_feat'].apply(lambda x: x['skew'])
  merged_dist['B_skew'] = merged_dist['B_feat'].apply(lambda x: x['skew'])
  
  merged_dist['A_kurtosis'] = merged_dist['A_feat'].apply(lambda x: x['kurtosis'])
  merged_dist['B_kurtosis'] = merged_dist['B_feat'].apply(lambda x: x['kurtosis'])
  
  merged_dist['A_dip'] = merged_dist['A_feat'].apply(lambda x: x['dip'])
  merged_dist['B_dip'] = merged_dist['B_feat'].apply(lambda x: x['dip'])

  merged_dist['shape_distance'] = (
    abs(merged_dist['A_skew'] - merged_dist['B_skew']) +
    abs(merged_dist['A_kurtosis'] - merged_dist['B_kurtosis']) +
    abs(merged_dist['A_dip'] - merged_dist['B_dip'])
  )
  
  merged_dist_ = merged_dist[[
    'A_var_name',
    'B_var_name',
    'A_skew',
    'B_skew',
    'A_kurtosis',
    'B_kurtosis',
    'A_dip',
    'B_dip',
    'shape_distance'
  ]]

  return merged_dist_

### Structured 06 - Measurement Unit Consistency

def clean_unit(u):
  if pd.isna(u):
      return None
  u = str(u).lower().strip()
  return u if u != '' else None

def extract_unit_set(df, var_name):
  return set(
      df.loc[
          (df['var_norm'] == var_name) &
          (df['unit_clean'].notna()),
          'unit_clean'
      ]
  )

def build_pair_unit_table(df_A, df_B, pair_df):

  results = []

  for _, row in pair_df.iterrows():

      var_A = row['A_var_name']
      var_B = row['B_var_name']

      A_units = extract_unit_set(df_A, var_A)
      B_units = extract_unit_set(df_B, var_B)

      results.append({
          'A_var_name': var_A,
          'B_var_name': var_B,
          'A_units': A_units,
          'B_units': B_units
      })

  return pd.DataFrame(results)

def safe_len(x):
  return len(x) if isinstance(x, set) else 0
  
def safe_union(a, b):
  if isinstance(a, set) and isinstance(b, set):
      return a.union(b)
  elif isinstance(a, set):
      return a
  elif isinstance(b, set):
      return b
  else:
      return set()

def compute_measurement_unit_consistency(df_A, df_B, pair_df) :
  quiq_a_ = df_A.copy()
  quiq_b_ = df_B.copy()

  quiq_a_['var_norm'] = quiq_a_['Variable_name'].apply(normalize)
  quiq_b_['var_norm'] = quiq_b_['Variable_name'].apply(normalize)
  quiq_a_['unit_clean'] = quiq_a_['Unit'].apply(clean_unit)
  quiq_b_['unit_clean'] = quiq_b_['Unit'].apply(clean_unit)

  merged = build_pair_unit_table(quiq_a_, quiq_b_, pair_df)

  merged['A_unit_count'] = merged['A_units'].apply(safe_len)
  merged['B_unit_count'] = merged['B_units'].apply(safe_len)

  merged['union_units'] = merged.apply(
    lambda row: safe_union(row['A_units'], row['B_units']),
    axis=1)

  merged['union_unit_count'] = merged['union_units'].apply(len)

  merged['is_mixed'] = merged['union_unit_count'] > 1

  result = merged[[
    'A_var_name',
    'B_var_name',
    'A_units', 'A_unit_count',
    'B_units', 'B_unit_count',
    'union_unit_count',
    'is_mixed'
  ]]

  return result

### Structured 07 - Medical Code Consistency

def sample_values(values, n=50, seed=42):

    values = (
        pd.Series(values)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(values) <= n:
        return values

    random.seed(seed)

    return random.sample(values, n)

def infer_variable_metadata_with_llm(
    client,
    model,
    table_name,
    variable_name,
    description,
    sample_values_list
):

    prompt = f"""
You are a medical terminology expert.

Your task:
Infer the most likely medical coding system and semantic type
for the variable below.

########################################################
Inference Rules

1. Semantic Type Inference
- Infer the semantic type primarily from:
  - original table name
  - variable name
  - variable description
- Do NOT rely only on the appearance of the codes.
- The same coding system may be used for different semantic types depending on context.
- Semantic types:
  - DIAGNOSIS
  - PROCEDURE
  - LAB
  - DRUG
  - OTHER

########################################################
2. Code System Inference
- Infer the coding system using:
  - metadata context
  - sample values
  - observed code patterns
- Allowed code systems:
  - ICD-9
  - ICD-10
  - CPT
  - HCPCS
  - LOINC
  - RXNORM
  - NDC
  - ATC
  - SNOMED
  - UNKNOWN

########################################################
3. Mixed Coding Systems
- mixed = true if multiple coding systems appear to coexist.
- dominant_code_system MUST contain exactly ONE primary system.
- Even if mixed=true, choose the MOST LIKELY or MOST FREQUENT primary system.
- Use UNKNOWN only if no reasonable dominant system can be inferred.
- secondary_code_systems should contain all additional systems except the dominant system.
- If mixed=false:
  - secondary_code_systems MUST be an empty list.

########################################################
4. General Principles
- Infer semantic type mainly from:
  - original table name
  - variable name
  - variable description
- Do NOT determine semantic type from code format alone.
- The same coding system can be used for different semantic meanings depending on context.
- Mixed coding systems may exist in the same variable.

########################################################
Output Requirements
- Output VALID JSON only.
- Do NOT provide explanations.
########################################################
Output format:

{{
  "semantic_type": "DIAGNOSIS",
  "dominant_code_system": "ICD-10",
  "mixed": true,
  "secondary_code_systems": ["ICD-9"]
}}

########################################################
Variable Information

Table Name:
{table_name}

Variable Name:
{variable_name}

Description:
{description}

Sample Values:
{sample_values_list}
"""

    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": prompt
        }]
    )

    text = response.output_text.strip()

    try:

        parsed = json.loads(text)

        return {
            "semantic_type": parsed.get(
                "semantic_type",
                "OTHER"
            ),

            "dominant_code_system": parsed.get(
                "dominant_code_system",
                "UNKNOWN"
            ),

            "mixed": parsed.get(
                "mixed",
                False
            ),

            "secondary_code_systems": parsed.get(
                "secondary_code_systems",
                []
            )
        }

    except:

        print("LLM parsing failure")
        print(text[:1000])

        return {
            "semantic_type": "OTHER",
            "dominant_code_system": "UNKNOWN",
            "mixed": False,
            "secondary_code_systems": []
        }

def build_variable_summary(
    quiq_df,
    via_df,
    center_name,
    client,
    model,
    sample_n=50
):

    """
    quiq_df columns:
        - Original_table_name
        - Variable_name
        - Value

    via_df columns:
        - Original_table_name
        - Variable_name
        - Description
    """

    variable_units = (
        quiq_df[
            ["Original_table_name", "Variable_name"]
        ]
        .drop_duplicates()
    )

    results = []

    for _, row in variable_units.iterrows():

        table_name = row["Original_table_name"]
        variable_name = row["Variable_name"]

        values = (
            quiq_df.loc[
                (
                    quiq_df["Original_table_name"]
                    == table_name
                )
                &
                (
                    quiq_df["Variable_name"]
                    == variable_name
                ),
                "Value"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        sampled_values = sample_values(
            values,
            n=sample_n
        )

        via_match = via_df.loc[
            (
                via_df["Original_table_name"]
                == table_name
            )
            &
            (
                via_df["Variable_name"]
                == variable_name
            )
        ]

        if len(via_match) > 0:

            description = (
                via_match.iloc[0]
                .get("Description", "")
            )

        else:

            description = ""

        inferred = infer_variable_metadata_with_llm(
            client=client,
            model=model,
            table_name=table_name,
            variable_name=variable_name,
            description=description,
            sample_values_list=sampled_values
        )

        results.append({

            "Center":
                center_name,

            "Original_table_name":
                table_name,

            "Variable_name":
                variable_name,

            "Semantic_Type":
                inferred["semantic_type"],

            "Mixed":
                inferred["mixed"],

            "Dominant_Code_System":
                inferred[
                    "dominant_code_system"
                ],

            "Secondary_Code_System":
                (
                    inferred[
                        "secondary_code_systems"
                    ]
                    if inferred["mixed"]
                    else []
                )
        })

    return pd.DataFrame(results)

def compute_medical_code_consistency(
    summary_A,
    summary_B
):

    def aggregate_systems(df, prefix):

        rows = []

        for semantic_type, subdf in (
            df.groupby("Semantic_Type")
        ):

            systems = set()

            for dominant in subdf[
                "Dominant_Code_System"
            ]:

                if (
                    isinstance(dominant, str)
                    and dominant != "UNKNOWN"
                ):

                    systems.add(dominant)

            for secondary in subdf[
                "Secondary_Code_System"
            ]:

                if isinstance(secondary, list):

                    systems.update(
                        [
                            x for x in secondary
                            if x != "UNKNOWN"
                        ]
                    )

            rows.append({

                "Semantic_Type":
                    semantic_type,

                f"{prefix}_code_system":
                    systems
            })

        return pd.DataFrame(rows)
      

    sem_A = aggregate_systems(
        summary_A,
        prefix="A"
    )

    sem_B = aggregate_systems(
        summary_B,
        prefix="B"
    )

    merged = pd.merge(
        sem_A,
        sem_B,
        on="Semantic_Type",
        how="outer"
    )

    merged["A_code_system"] = (
        merged["A_code_system"]
        .apply(
            lambda x:
            x if isinstance(x, set)
            else set()
        )
    )

    merged["B_code_system"] = (
        merged["B_code_system"]
        .apply(
            lambda x:
            x if isinstance(x, set)
            else set()
        )
    )

    merged["Is_mismatch"] = (
        merged["A_code_system"] !=
        merged["B_code_system"]
    )

    return merged


def run_pipeline_medical_code(
    df_A,
    df_B,
    via_A,
    via_B,
    client,
    model,
    sample_n=50
):

    summary_A = build_variable_summary(
        quiq_df=df_A,
        center_name="A",
        via_df=via_A,
        client=client,
        model=model,
        sample_n=sample_n
    )

    summary_B = build_variable_summary(
        quiq_df=df_B,
        center_name="B",
        via_df=via_B,
        client=client,
        model=model,
        sample_n=sample_n
    )


    variable_summary_df = pd.concat(
        [summary_A, summary_B],
        ignore_index=True
    )

    consistency_df = compute_medical_code_consistency(
        summary_A,
        summary_B
    )

    return (
        variable_summary_df,
        consistency_df
    )
  

if __name__ == '__main__' :
  print('<LYDUS - Structured Data>\n')

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type = str)
  args = parser.parse_args()

  with open(args.config, 'r', encoding = 'utf-8') as file :
    config = yaml.safe_load(file)

  quiq_a_path = config.get('quiq_a_path')
  quiq_a = pd.read_csv(quiq_a_path, low_memory = False)
  quiq_b_path = config.get('quiq_b_path')
  quiq_b = pd.read_csv(quiq_b_path, low_memory = False)
  via_a_path = config.get('via_a_path')
  via_a = pd.read_csv(via_a_path, low_memory = False)
  via_b_path = config.get('via_b_path')
  via_b = pd.read_csv(via_b_path, low_memory = False)
  model_ver = config.get('model_ver')
  api_key = config.get('api_key')
  client = openai.OpenAI(api_key=api_key)
  save_path = config.get('save_path')

  print('Structured 01 - Variable Name Consistency')
  df_A_categorical = quiq_a.loc[quiq_a['Is_categorical'] == 1].copy()
  df_B_categorical = quiq_b.loc[quiq_b['Is_categorical'] == 1].copy()
  pair_df_categorical = run_full_pipeline(df_A_categorical, df_B_categorical, client, model_ver)

  df_A_continuous = quiq_a.loc[quiq_a['Is_categorical'] == 0].copy()
  df_B_continuous = quiq_b.loc[quiq_b['Is_categorical'] == 0].copy()
  pair_df_continuous = run_full_pipeline(df_A_continuous, df_B_continuous, client, model_ver)

  df_A_unit = quiq_a.loc[(quiq_a['Unit'].notna()) & (quiq_a['Unit'].astype(str).str.strip() != '')].copy()
  df_B_unit = quiq_b.loc[(quiq_b['Unit'].notna()) & (quiq_b['Unit'].astype(str).str.strip() != '')].copy()
  pair_df_unit = run_full_pipeline(df_A_unit, df_B_unit, client, model_ver)

  pair_df = pd.concat([pair_df_categorical, pair_df_continuous, pair_df_unit], axis = 0)
  pair_df = pair_df.drop_duplicates(subset = 'A_var_name', keep = 'first')
  pair_df = pair_df.drop_duplicates(subset = 'B_var_name', keep = 'first')
  
  variable_name_consistency = (1- pair_df['Is_hetero'].mean()) * 100
  pair_df.reset_index(drop = True).to_csv(save_path + '/01_Variable_name_consistency.csv')
  print(f'{variable_name_consistency:.3f}\n')

  print('Structured 02 - Categorical Value Consistency')
  df_surface = compute_surface_mismatch_from_pairs(df_A_categorical, df_B_categorical, pair_df_categorical, client, model_ver)
  categorical_value_consistency = df_surface['surface_match_rate'].mean()
  df_surface.reset_index(drop = True).to_csv(save_path  + '/02_Categorical_value_consistency.csv')
  print(f'{categorical_value_consistency:.3f}\n')

  print('Structured 03 - Categorical Variable Distribution Homogeneity')
  df_cat_dist = compute_categorical_distribution_heterogeneity(df_A_categorical, df_B_categorical, df_surface)
  categorical_variable_distribution_mean = df_cat_dist['js_divergence'].mean()
  categorical_variable_distribution_std = df_cat_dist['js_divergence'].std()
  df_cat_dist.reset_index(drop = True).to_csv(save_path  + '/03_Categorical_value_distribution_homogeneity.csv')
  print(f'mean: {categorical_variable_distribution_mean:.3f}\n')
  print(f'std: {categorical_variable_distribution_std:.3f}\n')

  print('Structured 04 - Continuous Variable Distribution Homogeneity')
  merged_dist = compute_continuous_variable_distribution_homogeniety(df_A_continuous, df_B_continuous, pair_df_continuous)
  continuous_variable_distribution_mean = merged_dist['wd_normalized'].mean()
  continuous_variable_distribution_std = merged_dist['wd_normalized'].std()
  merged_dist.reset_index(drop = True).to_csv(save_path  + '/04_Continuous_variable_distribution_homogeniety.csv')
  print(f'mean: {continuous_variable_distribution_mean:.3f}\n')
  print(f'std: {continuous_variable_distribution_std:.3f}\n')

  print('Structured 05 - Continuous Variable Distribution Shape Consistency')
  merged_dist_2 = compute_continuous_variable_distribution_shape_consistency(df_A_continuous, df_B_continuous, pair_df_continuous)
  continuous_variable_distribution_shape_mean = merged_dist_2['shape_distance'].mean()
  continuous_variable_distribution_shape_std = merged_dist_2['shape_distance'].std()
  merged_dist_2.reset_index(drop = True).to_csv(save_path + '/05_Continuous_variable_distribution_shape_consistency.csv')
  print(f'mean: {continuous_variable_distribution_shape_mean:.3f}\n')
  print(f'std: {continuous_variable_distribution_shape_std:.3f}\n')

  print('Structured 06 - Measurement Unit Consistency')
  result_unit = compute_measurement_unit_consistency(df_A_unit, df_B_unit, pair_df_unit)
  measurement_unit_consistency = (1 - result_unit['is_mixed'].mean()) * 100
  result_unit.reset_index(drop = True).to_csv(save_path + '/06_Measurement_unit_consistency.csv')
  print(f'{measurement_unit_consistency:.3f}\n')

  print('Structured 07 - Medical Code Consistency')
  df_A_medical_code = quiq_a.loc[quiq_a['Mapping_info_1'] == 'medical_code'].copy()
  df_B_medical_code = quiq_b.loc[quiq_b['Mapping_info_1'] == 'medical_code'].copy()
  variable_summary_df, consistency_df = run_pipeline_medical_code(df_A_medical_code, df_B_medical_code, via_A, via_B, client, model_ver, sample_n=50)
  medical_code_consistency = (1 - consistency_df['Is_mismatch'].mean()) * 100
  variable_summary_df.reset_index(drop = True).to_csv(save_path + '/07_Medical_code_consistency_detail.csv')
  consistency_df.reset_index(drop = True).to_csv(save_path + '/07_Medical_code_consistency.csv')
  print(f'{medical_code_consistency:.3f}\n')

  with open(save_path + '/LYDUS_multi_structured_total_results.txt', 'w', encoding = 'utf-8') as file :
    file.write(f'<LYDUS - Multi Structured>\n')
    file.write(f'01 Variable Name Consistency (%) = {variable_name_consistency:.2f}\n')
    file.write(f'02 Categorical Value Consistency (%) = {categorical_value_consistency:.2f}\n')
    file.write(f'03 Categorical Variable Distribution Homogeneity - mean = {categorical_variable_distribution_mean:.2f}\n')
    file.write(f'03 Categorical Variable Distribution Homogeneity - std = {categorical_variable_distribution_std:.2f}\n')
    file.write(f'04 Continuous Variable Distribution Homogeneity - mean = {continuous_variable_distribution_mean:.2f}\n')
    file.write(f'04 Continuous Variable Distribution Homogeneity - std = {continuous_variable_distribution_std:.2f}\n')
    file.write(f'05 Continuous Variable Distribution Shape Consistency - mean = {continuous_variable_distribution_shape_mean:.2f}\n')
    file.write(f'05 Continuous Variable Distribution Shape Consistency - std = {continuous_variable_distribution_shape_std:.2f}\n')
    file.write(f'06 Measurement Unit Consistency (%) = {measurement_unit_consistency:.2f}\n')
    file.write(f'07 Medical Code Consistency (%) = {medical_code_consistency:.2f}')

  print('SUCCESS')

