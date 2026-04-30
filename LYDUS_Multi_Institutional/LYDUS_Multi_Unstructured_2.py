"""
노트 텍스트 길이 이질성 분석 (B-2)
사용법:
    python b2_analysis.py
    python b2_analysis.py --config my_config.yaml
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import warnings
from itertools import combinations
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────
# 1. Config 로드
# ──────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f'[설정] {path} 로드 완료')
    for k, v in cfg.items():
        print(f'       {k:<12}: {v}')
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
# 3. B-2: 텍스트 길이 이질성
# ──────────────────────────────────────────
def run_b2(note_df: pd.DataFrame, min_n_wd: int) -> pd.DataFrame:
    print('=' * 60)
    print('B-2: 텍스트 길이 이질성 (Wasserstein distance)')
    print('=' * 60)

    b2_rows = []

    for note_type, grp in note_df.groupby('Note_type'):
        sites = grp['site'].unique()
        if len(sites) < 2:
            continue

        print(f'\n{"-" * 55}')
        print(f'노트 유형: {note_type}')
        print(f'{"-" * 55}')

        for s1, s2 in combinations(sites, 2):
            la = grp[grp['site'] == s1]['Token_count'].values
            lb = grp[grp['site'] == s2]['Token_count'].values

            if len(la) < min_n_wd or len(lb) < min_n_wd:
                print(f'\n  [SKIP] {s1} ↔ {s2}: 문서 수 부족 ({len(la)}건 / {len(lb)}건, 최소 {min_n_wd}건)')
                continue

            print(f'\n  비교: {s1}  ↔  {s2}')
            print(f'  Step 1  길이 분포 계산')
            print(f'          {s1:<15}  건수={len(la):>4}  최소={int(la.min()):>6}자  중앙값={int(np.median(la)):>6}자  평균={int(la.mean()):>6}자  최대={int(la.max()):>6}자')
            print(f'          {s2:<15}  건수={len(lb):>4}  최소={int(lb.min()):>6}자  중앙값={int(np.median(lb)):>6}자  평균={int(lb.mean()):>6}자  최대={int(lb.max()):>6}자')

            wd = wasserstein_distance(la, lb)
            print(f'  Step 2  Wasserstein distance = {wd:.1f}자')
            print(f'          해석: 평균적으로 {wd:.0f}자만큼 두 기관 노트 길이가 다름')

            b2_rows.append({
                'Note_type':            note_type,
                'Site_A':               s1,
                'Site_B':               s2,
                'N_A':                  len(la),
                'N_B':                  len(lb),
                'Median_A':             int(np.median(la)),
                'Median_B':             int(np.median(lb)),
                'Wasserstein_distance': round(wd, 2),
            })

    return pd.DataFrame(b2_rows)


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

    b2 = run_b2(note_df, min_n_wd=cfg['min_n_wd'])

    print('\n' + '=' * 60)
    print('B-2 최종 결과')
    print('=' * 60)

    if b2.empty:
        print('[결과 없음] 유효한 노트 유형이 없습니다.')
    else:
        print(b2[['Note_type', 'Site_A', 'Site_B', 'Median_A', 'Median_B',
                   'Wasserstein_distance', 'Interpretation']].to_string(index=False))
        b2.to_csv(cfg['out_b2'], index=False, encoding='utf-8-sig')
        print(f'\n결과 저장 완료: {cfg["out_b2"]}')


if __name__ == '__main__':
    main()
