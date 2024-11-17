import pandas as pd


# TODO: Implement the logic to actually iterate over the labeled data and improve the prompt - Add any other input params as needed
def ImprovePrompt(current_prompt: dict, all_labeled_data: pd.DataFrame, misclassified: pd.DataFrame, classification_request: str, classes: list) -> dict:
    """
    This function is responsible for improving the prompt by iterating over the labeled data and improving the prompt
    It uses the miss-classified notes as few shots to improve the classification prompt.
    """
    # For now, we will just return the current prompt
    return current_prompt
    