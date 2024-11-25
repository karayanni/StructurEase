import asyncio
import pandas as pd
import logging
import time
import random
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


async def extract_classification_number(response: str, system_prompt: str):
    """
    This function uses the LLM to extract the classification number from the given response.
    It ensures that the extracted result is only the digit representing the classification.
    """
    client = AsyncOpenAI()

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
    async def make_completion_request():
        curr_completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            top_p=1,
            logprobs=True,
            messages=message_list
        )
        return curr_completion

    # Use exponential backoff to retry the API call
    completion = exponential_backoff_retry(make_completion_request)
    extracted_number = completion.choices[0].message.content.strip()

    # Ensure the result is a valid digit
    if not extracted_number.isdigit():
        logging.error(f"Failed to extract a valid number from the response: {extracted_number}")
        return 0

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


async def exponential_backoff_retry(func, max_retries=4, initial_delay=1, max_delay=16, jitter=0.5):
    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            return await func()
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                logging.error(f"Function failed after {max_retries} retries: {e}")
                raise
            # Apply jitter to avoid thundering herd problem
            time.sleep(delay + random.uniform(0, jitter))
            delay = min(max_delay, delay * 2)
            logging.error(f"Retrying after {delay} seconds... (Attempt {retries}/{max_retries})")


async def classification_using_llm(clinical_note: str, system_prompt: str, user_prompt: str):
    # Initialize the OpenAI client
    client = AsyncOpenAI()
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
    async def make_completion_request():
        curr_completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            top_p=1,
            logprobs=True,
            messages=message_list
        )
        return curr_completion

    # Use exponential backoff to retry the API call in case of rate limits or transient errors
    completion = await exponential_backoff_retry(make_completion_request)
    assistant_response_content = completion.choices[0].message.content.strip()

    return assistant_response_content, completion


async def process_clinical_note_sync(clinical_note, system_prompt: str, user_prompt: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        assistant_response_content, completion = await classification_using_llm(clinical_note, system_prompt,
                                                                                user_prompt)
        # Extract the last character, which should be 0, 1, or 2
        last_char = assistant_response_content.strip()[-1]

        if not last_char.isdigit():
            logging.error(
                f"Unexpected response format. Response: {assistant_response_content}, System Prompt: {system_prompt}, User Prompt: {user_prompt}")
            print(f"Error: Unexpected response format. Response: {assistant_response_content}")
            print(f"System Prompt: {system_prompt}")
            print(f"User Prompt: {user_prompt}")
            return assistant_response_content, extract_classification_number, None  # Return None to indicate an invalid response

        return assistant_response_content, last_char, completion


def evaluate_classification_accuracy_on_entire_DS(system_prompt: str, user_prompt: str, output_file: str):
    """
    Evaluate the classification accuracy of LLM's output in the given prompts.
    """
    labeled_data_file = 'Evaluation/NEISS data/neiss_2023_filtered_2000_rows_labeled_mapped_fixed.csv'
    df = pd.read_csv(labeled_data_file)

    # df = df.head(1)

    responses = []
    numbers = []

    # Create or retrieve an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    semaphore = asyncio.Semaphore(200)

    tasks = []

    for i, row in df.iterrows():
        tasks.append(process_clinical_note_sync(row['Narrative_1'], system_prompt, user_prompt, semaphore))

    async def process_all_rows():
        return await asyncio.gather(*tasks)

    # Run the asynchronous tasks
    results_curr_prompt = loop.run_until_complete(process_all_rows())

    for (assistant_response_content, last_char, completion) in results_curr_prompt:
        responses.append(assistant_response_content)
        numbers.append(last_char)

    # Add the new columns to the DataFrame
    df['LLM_output'] = responses
    df['LLM_number'] = numbers

    df = df.dropna(subset=['Helmet_Status_Num'])

    # Convert columns to integers for accurate comparison
    df['Helmet_Status_Num'] = df['Helmet_Status_Num'].astype(int)
    df['LLM_number'] = df['LLM_number'].astype(int)

    # Compare Helmet_Status_Num with LLM_number
    df['Correct'] = df['Helmet_Status_Num'] == df['LLM_number']

    # save the output to a file
    df.to_csv(f"Evaluation/experiment_results/k200SamplingP1/{output_file}.csv", index=False)

    # Calculate accuracy
    total_cases = len(df)
    correct_cases = df['Correct'].sum()
    accuracy_rate = correct_cases / total_cases * 100

    # Extract misclassified rows for reporting
    misclassified = df[~df['Correct']].copy()

    # TODO: Add F1, RECALL AND ALL THAT SHIT...
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


def run_classification_on_entire_DS(df: pd.DataFrame, column_name: str, system_prompt: str, user_prompt: str, output_file: str):
    """
    Evaluate the classification accuracy of LLM's output in the given prompts.
    """

    responses = []
    numbers = []

    # Create or retrieve an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    semaphore = asyncio.Semaphore(200)

    tasks = []

    for i, row in df.iterrows():
        tasks.append(process_clinical_note_sync(row[column_name], system_prompt, user_prompt, semaphore))

    async def process_all_rows():
        return await asyncio.gather(*tasks)

    # Run the asynchronous tasks
    results_curr_prompt = loop.run_until_complete(process_all_rows())

    for (assistant_response_content, last_char, completion) in results_curr_prompt:
        responses.append(assistant_response_content)
        numbers.append(last_char)

    # Add the new columns to the DataFrame
    df['LLM_output'] = responses
    df['LLM_number'] = numbers

    return df


if __name__ == '__main__':
    classification_report = evaluate_classification_accuracy_on_entire_DS(
        system_prompt="You are an expert in analyzing patient injury reports specializing in determining whether a patient was wearing a helmet during an accident. Your task is to analyze the data provided and determine if the patient was wearing a helmet, not wearing a helmet, or if it cannot be determined. Consider all relevant information in the data and only in the provided data. For example, if the text explicitly states that the patient was wearing a helmet, classify it as 'Helmet'. If it states that the patient was not wearing a helmet, classify it as 'No Helmet'. If the information is ambiguous or missing, classify it as 'cannot determine'. Provide your answer as Helmet, No Helmet, cannot determine and appropriate number 0 if the answer is Helmet, 1 if the answer is No Helmet and 2 if the answer is cannot determine.\n\n AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1",
        user_prompt="You are a medical data analyst. Your task is to read an unstructured text and classify it according to the given classes. Please analyze the following free text data provided and determine if the patient was wearing a helmet, not wearing a helmet, or if it cannot be determined. MAKE SURE TO END YOUR RESPONSE WITH THE NUMBER ONLY.\n\n AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1",
        output_file="stam-test")
