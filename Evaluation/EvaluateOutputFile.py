import pandas as pd


def evaluate_classification_accuracy(file_path):
    """
    Evaluate the classification accuracy of LLM's output in the given CSV file.
    Args:
    - file_path (str): Path to the CSV file containing the Helmet_Status and LLM_number columns.

    Returns:
    - A dictionary containing the accuracy rate and a DataFrame of misclassified rows.
    """

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Map Helmet_Status values to numeric labels
    helmet_status_map = {
        'No helmet': 0,
        'Helmet': 1,
        'Unknown (not mentioned)': 2,
        'Unknonw (not mentioned)': 2  # Correcting for potential misspelling
    }

    # Apply the mapping to the Helmet_Status column
    df['Helmet_Status_Num'] = df['Helmet_Status'].map(helmet_status_map)

    # Drop rows where Helmet_Status or LLM_number are NaN (if any exist)
    df = df.dropna(subset=['Helmet_Status_Num', 'LLM_number'])

    # Convert columns to integers for accurate comparison
    df['Helmet_Status_Num'] = df['Helmet_Status_Num'].astype(int)
    df['LLM_number'] = df['LLM_number'].astype(int)

    # Compare Helmet_Status_Num with LLM_number
    df['Correct'] = df['Helmet_Status_Num'] == df['LLM_number']

    # Calculate accuracy
    total_cases = len(df)
    correct_cases = df['Correct'].sum()
    accuracy_rate = correct_cases / total_cases * 100

    # Extract misclassified rows for reporting
    misclassified = df[~df['Correct']].copy()

    # Create the report
    report = {
        "accuracy_rate": accuracy_rate,
        "total_cases": total_cases,
        "correct_cases": correct_cases,
        "incorrect_cases": total_cases - correct_cases,
        "misclassified_cases": misclassified[['CPSC_Case_Number', 'Helmet_Status', 'LLM_number', 'Narrative_1']]
    }

    # Display results
    print(f"Accuracy Rate: {accuracy_rate:.2f}%")
    print(f"Total Cases: {total_cases}")
    print(f"Correctly Classified Cases: {correct_cases}")
    print(f"Incorrectly Classified Cases: {total_cases - correct_cases}")

    if not misclassified.empty:
        print("\nMisclassified Cases:")
        print(misclassified[['CPSC_Case_Number', 'Helmet_Status', 'LLM_number', 'Narrative_1']])
    else:
        print("\nNo misclassified cases found.")

    return report


if __name__ == '__main__':
    output_file_path = 'NEISS data/neiss_filtered_labeled_output.csv'
    classification_report = evaluate_classification_accuracy(output_file_path)
