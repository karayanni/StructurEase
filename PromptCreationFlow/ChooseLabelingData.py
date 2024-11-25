import math

import pandas as pd
import logging
import time
import random
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio

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
        "incorrect_cases": total_cases - correct_cases
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
    # Placeholder for the actual processing logic
    # todo: remove when we need to call the LLM.
    # return "test test text", 2
    async with semaphore:
        assistant_response_content, completion = await classification_using_llm(clinical_note, system_prompt, user_prompt)
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


def ChooseLabelingData(df: pd.DataFrame, column_name: str, current_prompt: dict, already_chosen_data_indices: list[int]):
    """
    This function chooses the next K data rows for manual labeling.
    """
    # Filter out rows with indices in already_chosen_data_indices
    filtered_df = df.drop(index=already_chosen_data_indices, errors='ignore')

    # Sample data from the filtered DataFrame
    sampled_data = filtered_df[column_name].sample(200)

    # Create or retrieve an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Explicitly tie the semaphore to the event loop
    semaphore = asyncio.Semaphore(100)

    tasks = []

    for i, row in sampled_data.items():
        print(f"Index: {i}, Text: {row}")
        tasks.append(process_clinical_note_sync(row, current_prompt['system_message'], current_prompt['user_message'], semaphore))

    async def process_all_rows():
        return await asyncio.gather(*tasks)

    # Run the asynchronous tasks
    results_curr_prompt = loop.run_until_complete(process_all_rows())

    # Process the results
    final_results = []
    for simple_index, (i, row) in enumerate(sampled_data.items()):
        assistant_response_content, last_char, completion = results_curr_prompt[simple_index]

        # Calculate confidence
        confidence = 1.0
        if completion:
            logprobs = [token.logprob for token in completion.choices[0].logprobs.content]
            confidence = math.exp(sum(logprobs) / len(logprobs))
        final_results.append((i, row, assistant_response_content, last_char, completion, confidence))

    # Sort results by confidence
    final_results.sort(key=lambda x: x[-1])

    # Choose up to 10 samples per class
    sampled_indices = []
    sampled_values = []
    samples_per_class = {'0': [], '1': [], '2': []}

    for i, row, assistant_response_content, last_char, completion, confidence in final_results:
        if len(samples_per_class[last_char]) >= 10:
            continue
        samples_per_class[last_char].append(i)
        sampled_indices.append(i)
        sampled_values.append(row)

    return sampled_indices, sampled_values


def ChooseLabelingDataRandom(df: pd.DataFrame, column_name: str, current_prompt: dict, already_chosen_data_indices: list[int]):
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

    sampled_data = filtered_df[column_name].sample(30)

    # Extract indices and data as separate outputs
    sampled_indices = sampled_data.index.tolist()
    sampled_values = sampled_data.values.tolist()

    return sampled_indices, sampled_values