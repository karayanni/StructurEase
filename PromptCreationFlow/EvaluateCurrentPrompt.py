import pandas as pd
import logging
import time
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


async def extract_classification_number(response: str, system_prompt: str):
    """
    This function uses the LLM to extract the classification number from the given response.
    It ensures that the extracted result is only the digit representing the classification.
    """
    client = OpenAI()

    extraction_prompt = (
        f"You are an expert at extracting structured information from free text. "
        f"The following text is a response where the classification number was expected to be the last character but wasn't. "
        f"Your task is to extract only the classification number (0, 1, or 2) from the context of the provided text. "
        f"Here is the response: \"{response}\". "
        f"Provide your answer as just the number without any explanation or additional text."
    )

    system_message = {
        "role": "system",
        "content": system_prompt
    }

    user_message = {
        "role": "user",
        "content": extraction_prompt
    }

    message_list = [system_message, user_message]

    # Function to make the API call
    def make_completion_request():
        curr_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            top_p=1,
            messages=message_list
        )
        return curr_completion

    # Use exponential backoff to retry the API call
    completion = exponential_backoff_retry(make_completion_request)
    extracted_number = completion.choices[0].message.content.strip()

    # Ensure the result is a valid digit
    if not extracted_number.isdigit():
        logging.error(f"Failed to extract a valid number from the response: {extracted_number}")
        extracted_number = extract_classification_number(response, system_prompt)

    return extracted_number


def evaluate_manual_to_llm_df(llm_df: pd.DataFrame, classes: list):
    """
    Evaluate the classification accuracy of LLM's output in the given CSV file.
    Args:
    - file_path (str): Path to the CSV file containing the Helmet_Status and LLM_number columns.

    Returns:
    - A dictionary containing the accuracy rate and a DataFrame of misclassified rows.
    """

    helmet_status_map = {}
    # Map Helmet_Status values to numeric labels
    for i, cls in enumerate(classes):
        helmet_status_map[cls] = i

    # Apply the mapping to the Helmet_Status column
    llm_df['Helmet_Status_Num'] = llm_df['Label'].map(helmet_status_map)

    # Drop rows where Helmet_Status or LLM_number are NaN (if any exist)
    df = llm_df.dropna(subset=['Helmet_Status_Num', 'LLM_number'])

    # Convert columns to integers for accurate comparison
    df['Helmet_Status_Num'] = df['Helmet_Status_Num'].astype(int)
    df['LLM_number'] = df['LLM_number'].astype(int)

    # Compare Helmet_Status_Num with LLM_number
    df['Correct'] = df['Helmet_Status_Num'] == df['LLM_number']

    correct_prediction_class = 0
    total_class_observations = 0
    total_class_predictions = 0

    for class_label in df['LLM_number'].unique():
        correct_prediction_class += \
        (df.loc[(df['LLM_number'] == class_label) & (df['Helmet_Status_Num'] == df['LLM_number'])]).shape[0]  # tp
        total_class_observations += (df['Helmet_Status_Num'] == class_label).sum()  # tp + fn
        total_class_predictions += (df['LLM_number'] == class_label).sum()  # tp + fp

    precision = correct_prediction_class / total_class_predictions
    recall = correct_prediction_class / total_class_observations
    macro_f1_score = 2 * ((precision * recall) / (precision + recall))

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
        "precision": precision,
        "recall": recall,
        "macro_f1_score": macro_f1_score
    }

    # Display results
    print(f"Accuracy Rate: {accuracy_rate:.2f}%")
    print(f"Total Cases: {total_cases}")
    print(f"Correctly Classified Cases: {correct_cases}")
    print(f"Incorrectly Classified Cases: {total_cases - correct_cases}")

    return report, misclassified[['Sampled Text', 'Label', 'LLM_number']]


def exponential_backoff_retry(func, max_retries=4, initial_delay=1, max_delay=16, jitter=0.5):
    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logging.error(f"Function failed after {max_retries} retries: {e}")
                raise
            # Apply jitter to avoid thundering herd problem
            time.sleep(delay + random.uniform(0, jitter))
            delay = min(max_delay, delay * 2)
            logging.error(f"Retrying after {delay} seconds... (Attempt {retries}/{max_retries})")


def classification_using_llm(clinical_note: str, system_prompt: str, user_prompt: str):
    # Initialize the OpenAI client
    client = OpenAI()
    message_list = []

    # todo: consider adding a wrapper to the provided prompt to ensure the LLM returns a valid response ending with a number.
    ensure_end_in_number = f"\n\n MAKE SURE YOU END YOUR RESPONSE WITH THE NUMBER THAT CORRESPONDS THE THE CLASS AND ONLY WITH THE NUMBER - MAKE SURE THE LAST CHARACTER OF YOUR RESPONSE IS THE NUMBER ONLY. \n\n"

    system_message = {
        "role": "system",
        "content": system_prompt
    }

    user_message = {
        "role": "user",
        "content": user_prompt + ensure_end_in_number + f"\n\n Here is the Clinical Note: \n\n \"{clinical_note}\""

    }

    # Add messages to the conversation
    message_list.append(system_message)
    message_list.append(user_message)

    # Function to make the API call
    def make_completion_request():
        curr_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            top_p=1,
            messages=message_list
        )
        return curr_completion

    # Use exponential backoff to retry the API call in case of rate limits or transient errors
    completion = exponential_backoff_retry(make_completion_request)
    assistant_response = completion.choices[0].message.content.strip()

    return assistant_response


def process_clinical_note_sync(clinical_note, system_prompt: str, user_prompt: str):
    # Placeholder for the actual processing logic
    # todo: remove when we need to call the LLM.
    # return "test test text", 2

    response = classification_using_llm(clinical_note, system_prompt, user_prompt)
    # Extract the last character, which should be 0, 1, or 2
    last_char = response.strip()[-1]

    if not last_char.isdigit():
        logging.error(
            f"Unexpected response format. Response: {response}, System Prompt: {system_prompt}, User Prompt: {user_prompt}")
        print(f"Error: Unexpected response format. Response: {response}")
        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt: {user_prompt}")
        return response, None  # Return None to indicate an invalid response

    return response, last_char


def EvaluateCurrentPrompt(system_prompt: str, user_prompt: str, labeled_data: pd.DataFrame, classes: list):
    """
    This function reads the NEISS data, processes the clinical notes, and classifies the helmet status using the provided prompts.
    :param system_prompt: The system prompt to instruct the assistant on how to classify the helmet status.
    :param user_prompt:  The user prompt to provide context and request the classification of the helmet status.
    :return: Saves the processed data to a new CSV file.
    """
    df_processed = labeled_data.copy()

    responses = []
    numbers = []

    for idx, row in labeled_data.iterrows():
        # Process the clinical note
        response, number = process_clinical_note_sync(row['Sampled Text'], system_prompt, user_prompt)
        responses.append(response)
        numbers.append(number)

    # Add the new columns to the DataFrame
    df_processed['LLM_output'] = responses
    df_processed['LLM_number'] = numbers

    return evaluate_manual_to_llm_df(df_processed, classes)


if __name__ == '__main__':
    # todo: test this and finalize...
    print("This is the EvaluateCurrentPrompt.py file.")
    