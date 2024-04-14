import pandas as pd
import sys  # Import the sys module

def main(csv_path):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_path, low_memory=False)

    # 데이터셋 전체의 완성도 비율 계산
    total_non_missing_values = df['Value'].count()
    total_observations = len(df)
    overall_completeness_ratio = (total_non_missing_values / total_observations) * 100

    print(f"완전성: {overall_completeness_ratio:.2f}%")

    # 세부내용에 '변수명'별 완성도 비율을 출력합니다.
    non_null_counts_per_variable = df.groupby('Variable_name')['Value'].count()
    total_counts_per_variable = df.groupby('Variable_name').size()
    completeness_ratio_per_variable = (non_null_counts_per_variable / total_counts_per_variable) * 100

    completeness_df = pd.DataFrame({
        '변수명': completeness_ratio_per_variable.index,
        '완전성 (%)': completeness_ratio_per_variable.values.round(3)  # Round to 3 decimal places
    })

    print(completeness_df)

    # Save the completeness details to a CSV file
    completeness_df.to_csv('결과_완전성.csv', index=False)
    print("Detailed results are saved as '결과_완전성.csv'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the CSV file path as a command line argument.")
        sys.exit(1)  # Exit the script with an error code

    csv_path = sys.argv[1]  # Get the CSV file path provided as a command line argument
    main(csv_path)
