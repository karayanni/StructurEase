import pandas as pd


def ChooseLabelingData(df: pd.DataFrame, column_name: str, current_prompt: dict, already_chosen_data_indices: list[int]):
    """
    This function is used to choose the next K data rows that will be used for manual labeling.
    The function will return a list of indices of the chosen data rows. It uses the current prompt to assess diverse
    and challenging rows for manual labeling to maximize the model's learning form the human input.
    :param df: DataFrame, the input data
    :param column_name: str, the column name of the data to be labeled
    :param current_prompt: str, the latest prompt used to classify the data
    :param already_chosen_data_indices: list, indices of the data rows that have already been chosen for manual labeling
    :return: list, indices of the chosen data rows
    """
    # Filter out rows with indices in already_chosen_data_indices
    filtered_df = df.drop(index=already_chosen_data_indices, errors='ignore')

    # todo: instead of sampling 10 rows randomly, run a first pass of the model to get the most 'uncertain'ish rows
    sampled_data = filtered_df[column_name].sample(10, random_state=42)

    # Extract indices and data as separate outputs
    sampled_indices = sampled_data.index.tolist()
    sampled_values = sampled_data.values.tolist()

    return sampled_indices, sampled_values
