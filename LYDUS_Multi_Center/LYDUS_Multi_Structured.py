import pandas as pd
import numpy as np
import openai
import json
import re
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
  4. Each variable in A can match AT MOST one variable in B.
  5. Each variable in B can match AT MOST one variable in A.
  6. If no match exists, do not include it.
  7. Be conservative (avoid false matches).

  Output format:
  [
    {{"A": "blood sugar", "B": "glucose"}},
    {{"A": "heart rate", "B": "hr"}}
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

  pair_df = pair_df.drop_duplicates(subset='B', keep='first')
  pair_df = pair_df.drop_duplicates(subset='A', keep='first')

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
          is_hetero = None
          A_dist, B_dist = {}, {}

      else:
          def get_distribution(values, categories):
            counts = np.array([values.count(c) for c in categories])
            return counts / counts.sum()

          P = get_distribution(A_norm, common_categories)
          Q = get_distribution(B_norm, common_categories)

          jsd = js_divergence(P, Q)
          is_hetero = jsd > 0.1

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
          "Is_hetero": is_hetero
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

  

def compute_ks(a, b):
  if isinstance(a, list) and isinstance(b, list) and len(a) > 0 and len(b) > 0:
      stat, p = ks_2samp(a, b)
      return stat, p
  return np.nan, np.nan

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

  merged_dist[['ks_stat', 'ks_p']] = merged_dist.apply(
    lambda row: pd.Series(compute_ks(row['A_values'], row['B_values'])),
    axis=1
  )
  merged_dist['wasserstein_dist'] = merged_dist.apply(
      lambda row: compute_wd(row['A_values'], row['B_values']),
      axis=1
  )
  merged_dist['ks_hetero_flag'] = merged_dist['ks_p'] < 0.05

  merged_dist['wd_normalized'] = merged_dist.apply(normalize_wd, axis=1)
  merged_dist['wd_hetero_flag'] = merged_dist['wd_normalized'] > 0.5

  merged_dist_ = merged_dist.drop(['A_values', 'B_values'], axis=1)

  merged_dist_ = merged_dist_[[
      'A_var_name',
      'B_var_name',
      'ks_stat',
      'ks_p',
      'ks_hetero_flag',
      'wasserstein_dist',
      'wd_normalized',
      'wd_hetero_flag'
  ]]

  merged_dist_['Is_hetero'] = (
      merged_dist_['ks_hetero_flag'] &
      merged_dist_['wd_hetero_flag']
  )

  return merged_dist_

### Structured 05 - Continuous Variable Distribution Shape Consistency

def detect_distribution_shape(values):

  if not isinstance(values, list) or len(values) < 20:
      return "insufficient"

  values = np.array(values)

  try:
      dip, dip_p = diptest(values)
      if dip_p < 0.05:
          return "bimodal"
  except:
      pass

  # 2. normality
  try:
      stat, p = shapiro(values[:5000])
  except:
      return "unknown"

  sk = skew(values)
  kt = kurtosis(values)

  if p >= 0.05 and abs(sk) < 1 and abs(kt) < 3:
      return "normal"

  return "non_normal"


def compute_continuous_variable_distribution_shape_consistency(df_A, df_B, pair_df):
  quiq_a_ = df_A.copy()
  quiq_b_ = df_B.copy()

  quiq_a_['var_norm'] = quiq_a_['Variable_name'].apply(normalize)
  quiq_b_['var_norm'] = quiq_b_['Variable_name'].apply(normalize)
  quiq_a_['value_clean'] = quiq_a_['Value'].apply(clean_value)
  quiq_b_['value_clean'] = quiq_b_['Value'].apply(clean_value)

  merged_dist = build_pair_distribution(quiq_a_, quiq_b_, pair_df)

  merged_dist['A_shape'] = merged_dist['A_values'].apply(detect_distribution_shape)
  merged_dist['B_shape'] = merged_dist['B_values'].apply(detect_distribution_shape)

  # normal vs non-normal mismatch
  merged_dist['normality_mismatch'] = (
      ((merged_dist['A_shape'] == 'normal') & (merged_dist['B_shape'] != 'normal')) |
      ((merged_dist['A_shape'] != 'normal') & (merged_dist['B_shape'] == 'normal'))
  )

  # bimodal mismatch
  merged_dist['bimodal_mismatch'] = (
      ((merged_dist['A_shape'] == 'bimodal') & (merged_dist['B_shape'] != 'bimodal')) |
      ((merged_dist['A_shape'] != 'bimodal') & (merged_dist['B_shape'] == 'bimodal'))
  )

  # overall shape mismatch
  merged_dist['shape_mismatch'] = (
      merged_dist['A_shape'] != merged_dist['B_shape']
  )

  merged_dist_ = merged_dist[[
      'A_var_name',
      'B_var_name',
      'A_shape',
      'B_shape',
      'shape_mismatch'
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

def classify_code_system(code):
  code = str(code).strip().upper()

  # -------------------------
  # ICD-10 / KCD (알파벳 + 숫자)
  # -------------------------
  if re.match(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$', code):
      # KCD는 ICD-10 기반이라 heuristic 유지
      if '.' not in code and len(code) >= 4:
          return "KCD"
      return "ICD-10"

  # -------------------------
  # ICD-9 (숫자 기반)
  # -------------------------
  if re.match(r'^[0-9]{3}(\.[0-9]{1,2})?$', code):
      return "ICD-9"

  # -------------------------
  # CPT (5-digit numeric)
  # -------------------------
  if re.match(r'^[0-9]{5}$', code):
      return "CPT"

  # -------------------------
  # HCPCS (1 letter + 4 digits)
  # -------------------------
  if re.match(r'^[A-Z][0-9]{4}$', code):
      return "HCPCS"

  # -------------------------
  # LOINC (e.g., 1234-5)
  # -------------------------
  if re.match(r'^[0-9]{1,7}-[0-9]$', code):
      return "LOINC"

  # -------------------------
  # RxNorm (numeric, 보통 2~6 digits 이상)
  # -------------------------
  if re.match(r'^[0-9]{2,8}$', code):
      return "RXNORM"

  # -------------------------
  # NDC (10~11 digit, 또는 하이픈 포함)
  # -------------------------
  if re.match(r'^(\d{4,5}-\d{3,4}-\d{1,2}|\d{10,11})$', code):
      return "NDC"

  # -------------------------
  # ATC (e.g., A10BA02)
  # -------------------------
  if re.match(r'^[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}$', code):
      return "ATC"

  # -------------------------
  # SNOMED CT (numeric, usually long)
  # -------------------------
  if re.match(r'^[0-9]{6,18}$', code):
      return "SNOMED"

  return "UNKNOWN"

def map_semantic_from_code_system(code_system):

  diagnosis_systems = ["ICD-9", "ICD-10", "KCD", "SNOMED"]
  procedure_systems = ["CPT", "HCPCS"]
  lab_systems = ["LOINC"]
  drug_systems = ["RXNORM", "NDC", "ATC"]

  if code_system in diagnosis_systems:
      return "DIAGNOSIS"

  elif code_system in procedure_systems:
      return "PROCEDURE"

  elif code_system in lab_systems:
      return "LAB"

  elif code_system in drug_systems:
      return "DRUG"

  else:
      return "OTHER"

def summarize_code_system(df):

  df = df.copy()

  df["Code_System"] = df["Value"].apply(classify_code_system)
  df["Semantic_Type"] = df["Code_System"].apply(map_semantic_from_code_system)

  summary = (
      df.groupby(["Variable_name", "Semantic_Type"])["Code_System"]
      .agg(lambda x: set(x))
      .reset_index()
      .rename(columns={"Code_System": "code_system_set"})
  )

  return summary

def compute_medical_code_consistency(df_A, df_B):
  quiq_a_ = df_A.copy()
  quiq_b_ = df_B.copy()

  summary_A = summarize_code_system(quiq_a_)
  summary_B = summarize_code_system(quiq_b_)

  summary_sem_A = (
    summary_A.groupby("Semantic_Type")["code_system_set"]
    .agg(lambda x: set().union(*x))
    .reset_index()
  )

  summary_sem_B = (
      summary_B.groupby("Semantic_Type")["code_system_set"]
      .agg(lambda x: set().union(*x))
      .reset_index()
  )

  merged_sem = pd.merge(
      summary_sem_A,
      summary_sem_B,
      on="Semantic_Type",
      suffixes=("_A", "_B")
  )

  merged_sem["is_mismatch"] = (
      merged_sem["code_system_set_A"] != merged_sem["code_system_set_B"]
  )

  merged_sem = merged_sem.rename(columns={"code_system_set_A": "A_code_system",
                                        "code_system_set_B": "B_code_system",
                                        "is_mismatch": "Is_mismatch"})

  return merged_sem


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
  pair_df.reset_index(drop = True).to_csv(save_path + '/01 Variable_name_consistency.csv')
  print(f'{variable_name_consistency:.3f}\n')

  print('Structured 02 - Categorical Value Consistency')
  df_surface = compute_surface_mismatch_from_pairs(df_A_categorical, df_B_categorical, pair_df_categorical, client, model_ver)
  categorical_value_consistency = df_surface['surface_match_rate'].mean()
  df_surface.reset_index(drop = True).to_csv(save_path  + '/02 Categorical_value_consistency.csv')
  print(f'{categorical_value_consistency:.3f}\n')

  print('Structured 03 - Categorical Variable Distribution Homogeneity')
  df_cat_dist = compute_categorical_distribution_heterogeneity(df_A_categorical, df_B_categorical, df_surface)
  categorical_variable_distribution_homogeniety = (1 - df_cat_dist['Is_hetero'].mean()) * 100
  df_cat_dist.reset_index(drop = True).to_csv(save_path  + '/03 Categorical_value_distribution_homogeneity.csv')
  print(f'{categorical_variable_distribution_homogeniety:.3f}\n')

  print('Structured 04 - Continuous Variable Distribution Homogeneity')
  merged_dist = compute_continuous_variable_distribution_homogeniety(df_A_continuous, df_B_continuous, pair_df_continuous)
  continuous_variable_distribution_homogeniety = (1 - merged_dist['Is_hetero'].mean()) * 100
  merged_dist.reset_index(drop = True).to_csv(save_path  + '/04 Continuous_variable_distribution_homogeniety.csv')
  print(f'{continuous_variable_distribution_homogeniety:.3f}\n')

  print('Structured 05 - Continuous Variable Distribution Shape Consistency')
  merged_dist_2 = compute_continuous_variable_distribution_shape_consistency(df_A_continuous, df_B_continuous, pair_df_continuous)
  continuous_variable_distribution_shape_consistency = (1 - merged_dist['Is_hetero'].mean()) * 100
  merged_dist_2.reset_index(drop = True).to_csv(save_path + '/05 Continuous_variable_distribution_shape_consistency.csv')
  print(f'{continuous_variable_distribution_shape_consistency:.3f}\n')

  print('Structured 06 - Measurement Unit Consistency')
  result_unit = compute_measurement_unit_consistency(df_A_unit, df_B_unit, pair_df_unit)
  measurement_unit_consistency = (1 - result_unit['is_mixed'].mean()) * 100
  result_unit.reset_index(drop = True).to_csv(save_path + '/06 Measurement_unit_consistency.csv')
  print(f'{measurement_unit_consistency:.3f}\n')

  print('Structured 07 - Medical Code Consistency')
  df_A_medical_code = quiq_a.loc[quiq_a['Mapping_info_1'] == 'medical_code'].copy()
  df_B_medical_code = quiq_b.loc[quiq_b['Mapping_info_1'] == 'medical_code'].copy()
  merged_sem = compute_medical_code_consistency(df_A_medical_code, df_B_medical_code)
  medical_code_consistency = (1 - merged_sem['Is_mismatch'].mean()) * 100
  merged_sem.reset_index(drop = True).to_csv(save_path + '/07 Medical_code_consistency.csv')
  print(f'{medical_code_consistency:.3f}\n')

  with open(save_path + '/LYDUS_multi_structured_total_results.txt', 'w', encoding = 'utf-8') as file :
    file.write(f'<LYDUS - Multi Structured>\n')
    file.write(f'01 Variable Name Consistency (%) = {variable_name_consistency}\n')
    file.write(f'02 Categorical Value Consistency (%) = {categorical_value_consistency}\n')
    file.write(f'03 Categorical Variable Distribution Homogeneity (%) = {categorical_variable_distribution_homogeniety}\n')
    file.write(f'04 Continuous Variable Distribution Homogeneity (%) = {continuous_variable_distribution_homogeniety}\n')
    file.write(f'05 Continuous Variable Distribution Shape Consistency (%) = {continuous_variable_distribution_shape_consistency}\n')
    file.write(f'06 Measurement Unit Consistency (%) = {measurement_unit_consistency}\n')
    file.write(f'07 Medical Code Consistency (%) = {medical_code_consistency}')

  print('SUCCESS')

