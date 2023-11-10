import sys
import pandas as pd
import numpy as np


def compute_patient_fidelity(data, test_name):
    """
    Computes the fidelity for each patient for a given test.
    """
    test_data = data[data['변수명'] == test_name]
    patient_fidelity = {}

    for patient in test_data['환자 번호'].unique():
        patient_data = test_data[test_data['환자 번호'] == patient]

        # Check for non-empty values in the '값' column
        if patient_data['값'].notna().any():
            fidelity = patient_data['값'].notna().sum() / len(patient_data)
        else:
            fidelity = 0.0

        patient_fidelity[patient] = fidelity

    return patient_fidelity


def compute_test_fidelity(data):
    """
    Computes the average fidelity for each test.
    """
    average_fidelity_by_test = {}

    for test in data['변수명'].unique():
        patient_fidelities = compute_patient_fidelity(data, test)
        average_fidelity_by_test[test] = sum(patient_fidelities.values()) / len(patient_fidelities)

    return average_fidelity_by_test


def fidelity_summary_and_data(csv_path, print_details=False):
    """
    Calculates the overall average and standard deviation of fidelities and
    also returns the average fidelity for each test.
    """
    data = pd.read_csv(csv_path)

    avg_by_test = compute_test_fidelity(data)
    overall_avg = sum(avg_by_test.values()) / len(avg_by_test)
    overall_std = np.std(list(avg_by_test.values()))

    summary = f"전체 검사의 평균 충실도: {overall_avg:.3f}"

    if print_details:
        print(summary)
        print(f"전체 검사의 평균 충실도의 표준편차: {overall_std:.3f}")
        print("각 검사별로 계산한 평균 충실도:")
        for test, avg in avg_by_test.items():
            print(f"* {test} 검사: 평균 충실도 = {avg:.3f}")

    return summary, avg_by_test


if __name__ == "__main__":
    csv_path = sys.argv[1]  # 명령줄에서 제공된 CSV 파일 경로
    final_summary, avg_fidelity_by_test = fidelity_summary_and_data(csv_path, print_details=False)
    print(final_summary)