import pandas as pd


def evaluate_classification_accuracy(file1_path, file2_path):
    """
    Compare results from two NEISS files to evaluate classification accuracy.

    Args:
    - file1_path (str): Path to the first file (neiss_filtered_labeled_output).
    - file2_path (str): Path to the second file (neiss_2023_filtered_labeled).

    Returns:
    - A dictionary containing the accuracy rate and a DataFrame of misclassified rows.
    """
    # Load both CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Map Helmet_Status values to numeric labels for consistency
    helmet_status_map = {
        'No helmet': 0,
        'Helmet': 1,
        'Unknown (not mentioned)': 2,
        'Unknonw (not mentioned)': 2  # Correcting for potential misspelling
    }

    # Apply mapping to Helmet_Status columns in both files
    df1['Helmet_Status_Num'] = df1['LLM_number']
    df2['Helmet_Status_Num'] = df2['Helmet_Status'].map(helmet_status_map)

    # Ensure LLM_number is present in file1 and comparable
    if 'LLM_number' not in df1.columns:
        raise ValueError("Column 'LLM_number' is missing in the first file.")
    #
    # # Filter for common cases
    # common_cases = df1.index.intersection(df2.index)
    # df1 = df1.loc[common_cases]
    # df2 = df2.loc[common_cases]

    # Compare Helmet_Status_Num and LLM_number
    comparison_df = pd.DataFrame({
        "Treatment_Date_1": df1['Treatment_Date'],
        "Treatment_Date_2": df2['Treatment_Date'],
        "Helmet_Status_1": df1['Helmet_Status_Num'],
        "Helmet_Status_2": df2['Helmet_Status_Num'],
        "LLM_number": df1['LLM_number']
    })

    # Add comparison results
    comparison_df['Correct'] = comparison_df['Helmet_Status_1'] == comparison_df['Helmet_Status_2']

    # Calculate accuracy
    total_cases = len(comparison_df)
    correct_cases = comparison_df['Correct'].sum()
    accuracy_rate = correct_cases / total_cases * 100

    # Extract misclassified rows
    misclassified = comparison_df[~comparison_df['Correct']].copy()

    # Create report
    report = {
        "accuracy_rate": accuracy_rate,
        "total_cases": total_cases,
        "correct_cases": correct_cases,
        "incorrect_cases": total_cases - correct_cases,
        "misclassified_cases": misclassified
    }

    # Display results
    print(f"Accuracy Rate: {accuracy_rate:.2f}%")
    print(f"Total Cases: {total_cases}")
    print(f"Correctly Classified Cases: {correct_cases}")
    print(f"Incorrectly Classified Cases: {total_cases - correct_cases}")

    if not misclassified.empty:
        print("\nMisclassified Cases:")
        print(misclassified)
    else:
        print("\nNo misclassified cases found.")

    return report


if __name__ == '__main__':
    file1_path = 'NEISS data/neiss_filtered_labeled_output.csv'
    file2_path = 'NEISS data/neiss_2023_filtered_labeled.csv'
    classification_report = evaluate_classification_accuracy(file1_path, file2_path)
