"""
노트 스타일 분리도 분석 (B-1)
사용법:
    python note_style_analysis.py
    python note_style_analysis.py --config my_config.yaml
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────
# 1. Config 로드
# ──────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f'[설정] {path} 로드 완료')
    print(f'       q1      : {cfg["q1"]}')
    print(f'       q2      : {cfg["q2"]}')
    print(f'       out     : {cfg["out"]}')
    print(f'       sample  : {cfg["sample"]}')
    print(f'       min_n   : {cfg["min_n"]}')
    return cfg


# ──────────────────────────────────────────
# 2. QUIQ → 노트 DataFrame 생성
# ──────────────────────────────────────────
def make_note_df(q1: pd.DataFrame, q2: pd.DataFrame) -> pd.DataFrame:
    q1, q2 = q1.copy(), q2.copy()

    def get_site(df):
        v = df['Recorder_affiliation'].dropna().unique()
        return str(v[0]) if len(v) == 1 else 'unknown'

    q1['site'] = get_site(q1)
    q2['site'] = get_site(q2)

    df = pd.concat([q1, q2], ignore_index=True)

    df = df[
        df['Mapping_info_1'].isin(['note_clinical', 'note_rad']) &
        df['Value'].notna()
    ].copy()

    df['Text']        = df['Value'].astype(str)
    df['Token_count'] = df['Text'].str.len()
    df['Note_type']   = (
        df['Mapping_info_1'] + '_' + df['Mapping_info_2'].fillna('')
    ).str.rstrip('_')

    return df


# ──────────────────────────────────────────
# 3. B-1 분석
# ──────────────────────────────────────────
def run_b1(note_df: pd.DataFrame, max_sample: int, min_n: int) -> pd.DataFrame:
    print('=' * 60)
    print('B-1: 스타일 분리도 (TF-IDF + 5-fold CV 정확도)')
    print('=' * 60)

    print('\n[입력 데이터 요약]')
    summary = note_df.groupby(['site', 'Note_type']).agg(
        건수=('Text', 'count'),
        평균길이=('Token_count', 'mean'),
        중앙값길이=('Token_count', 'median')
    ).round(0)
    print(summary.to_string())

    b1_rows = []

    for note_type, grp in note_df.groupby('Note_type'):
        valid_sites = [
            s for s in grp['site'].unique()
            if len(grp[grp['site'] == s]) >= min_n
        ]
        if len(valid_sites) < 2:
            print(f'\n[SKIP] {note_type}: 유효 사이트 {len(valid_sites)}개 (최소 2개 필요)')
            continue

        print(f'\n{"-" * 55}')
        print(f'노트 유형: {note_type}')
        print(f'{"-" * 55}')

        sampled = pd.concat([
            grp[grp['site'] == s].sample(
                n=min(len(grp[grp['site'] == s]), max_sample),
                random_state=42
            )
            for s in valid_sites
        ], ignore_index=True)

        for s in valid_sites:
            n = len(sampled[sampled['site'] == s])
            print(f'  Step 1  샘플링  {s}: {n}건')

        X     = sampled['Text'].tolist()
        y     = sampled['site'].tolist()
        le    = LabelEncoder()
        y_enc = le.fit_transform(y)

        tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        X_vec = tfidf.fit_transform(X)
        print(f'  Step 2  TF-IDF 벡터화  →  {X_vec.shape[0]}개 노트 × {X_vec.shape[1]}개 n-gram')
        print(f'          레이블: {list(zip(le.classes_, range(len(le.classes_))))}')

        k      = min(5, pd.Series(y).value_counts().min())
        cv     = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        clf    = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, X_vec, y_enc, cv=cv, scoring='accuracy')

        print(f'  Step 3  {k}-fold CV 실행')
        for fold_i, s in enumerate(scores):
            bar = '█' * int(s * 20)
            print(f'          Fold {fold_i + 1}  |{bar:<20}|  {s:.3f}')

        clf.fit(X_vec, y_enc)

        mean_acc = float(np.mean(scores))
        std_acc  = float(np.std(scores))

        print(f'  {"-" * 45}')
        print(f'  CV 평균 정확도 : {mean_acc:.4f}')
        print(f'  CV 표준편차    : {std_acc:.4f}')

        b1_rows.append({
            'Note_type':         note_type,
            'Sites':             ' / '.join(sorted(valid_sites)),
            'CV_accuracy_mean':  round(mean_acc, 4),
            'CV_accuracy_std':   round(std_acc, 4),
            'Site_leakage_risk': mean_acc >= 0.85,
        })

    return pd.DataFrame(b1_rows)


# ──────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='설정 파일 경로')
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f'\n[로드] {cfg["q1"]}')
    quiq_1 = pd.read_csv(cfg['q1'])

    print(f'[로드] {cfg["q2"]}')
    quiq_2 = pd.read_csv(cfg['q2'])

    note_df = make_note_df(quiq_1, quiq_2)
    print(f'\n노트 행 수: {len(note_df):,}개')

    if note_df.empty:
        print('[ERROR] note_clinical / note_rad 에 해당하는 데이터가 없습니다.')
        print('        Mapping_info_1 고유값:', quiq_1['Mapping_info_1'].unique()[:10])
        return

    b1 = run_b1(note_df, max_sample=cfg['sample'], min_n=cfg['min_n'])

    print('\n' + '=' * 60)
    print('B-1 최종 결과')
    print('=' * 60)

    if b1.empty:
        print('[결과 없음] 유효한 노트 유형이 없습니다.')
    else:
        print(b1[['Note_type', 'CV_accuracy_mean', 'CV_accuracy_std', 'Interpretation']].to_string(index=False))
        b1.to_csv(cfg['out'], index=False, encoding='utf-8-sig')
        print(f'\n결과 저장 완료: {cfg["out"]}')


if __name__ == '__main__':
    main()
