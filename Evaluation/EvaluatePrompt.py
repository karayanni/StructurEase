import pandas as pd
import asyncio
import logging
import time
import random
from openai import AsyncOpenAI
from dotenv import load_dotenv
from Evaluation.EvaluateOutputFile import evaluate_classification_accuracy

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
            messages=message_list
        )
        return curr_completion

    # Use exponential backoff to retry the API call
    completion = await exponential_backoff_retry(make_completion_request)
    extracted_number = completion.choices[0].message.content.strip()

    # Ensure the result is a valid digit
    if not extracted_number.isdigit():
        logging.error(f"Failed to extract a valid number from the response: {extracted_number}")
        return None

    return extracted_number


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
            messages=message_list
        )
        return curr_completion

    # Use exponential backoff to retry the API call in case of rate limits or transient errors
    completion = await exponential_backoff_retry(make_completion_request)
    assistant_response = completion.choices[0].message.content.strip()

    return assistant_response


async def process_clinical_note(semaphore, row, system_prompt: str, user_prompt: str):
    # Placeholder for the actual processing logic
    # todo: remove when we need to call the LLM.
    # return "test test text", 2

    async with semaphore:
        clinical_note = row['Narrative_1']
        response = await classification_using_llm(clinical_note, system_prompt, user_prompt)
        # Extract the last character, which should be 0, 1, or 2
        last_char = response.strip()[-1]

        if not last_char.isdigit():
            logging.error(
                f"Unexpected response format. Response: {response}, System Prompt: {system_prompt}, User Prompt: {user_prompt}")
            print(f"Error: Unexpected response format. Response: {response}")
            print(f"System Prompt: {system_prompt}")
            print(f"User Prompt: {user_prompt}")

            extracted_number = await extract_classification_number(response, system_prompt)
            if extracted_number:
                last_char = extracted_number
            else:
                logging.error(f"Failed to extract the classification number from the response: {response}")
                return response, None

        return response, last_char


async def start_eval(system_prompt: str, user_prompt: str):
    """
    This function reads the NEISS data, processes the clinical notes, and classifies the helmet status using the provided prompts.
    :param system_prompt: The system prompt to instruct the assistant on how to classify the helmet status.
    :param user_prompt:  The user prompt to provide context and request the classification of the helmet status.
    :return: Saves the processed data to a new CSV file.
    """
    # Read the CSV file
    df = pd.read_csv('NEISS data/neiss_2023_filtered_labeled.csv')

    # Initialize a semaphore to limit concurrent API calls (adjust as needed)
    semaphore = asyncio.Semaphore(200)  # Limit to 100 concurrent tasks

    # Create a list of tasks for asynchronous execution
    tasks = []

    last_label_index = 0

    for idx, row in df.iterrows():
        tasks.append(process_clinical_note(semaphore, row, system_prompt, user_prompt))

    df_processed = df.iloc[:last_label_index].copy()

    # Execute the tasks and gather the results
    results = await asyncio.gather(*tasks)

    # Separate the assistant responses and the extracted numbers
    responses, numbers = zip(*results)

    # Add the new columns to the DataFrame
    df_processed['LLM_output'] = responses
    df_processed['LLM_number'] = numbers

    # Save the updated DataFrame to a new CSV file
    df_processed.to_csv('NEISS data/neiss_filtered_labeled_output.csv', index=False)
    print("Processing complete. Results saved to 'NEISS data/neiss_filtered_labeled_output.csv'.")


if __name__ == "__main__":
    system_prompt_message = (
        "You are an expert medical nurse specializing in injury assessment and emergency care. "
        "Your task is to analyze clinical notes and determine whether the injured person was wearing a helmet at the time of the injury. "
        "Consider all relevant information in the clinical note, including any mention of protective equipment, helmets, or lack thereof. "
        "HELMET+ OR +HELMET OR ANY OTHER INDICATOR OF HELMET SHOULD BE CONSIDERED AS A HELMET. "
        " -HELMET OR HELMET- OR ANY OTHER SIMILAR INDICATOR OF NO HELMET SHOULD BE CONSIDERED AS NO HELMET. "
        "If there is a mentions of NS or Helmet NS then this means Helmet Non-Specified and you should conclude 'Cannot determine'."
        "If the note explicitly states that the person was wearing a helmet, conclude 'Yes'. "
        "If the note explicitly states that the person was not wearing a helmet, conclude 'No'. "
        " If there is no mention of the patient wearing a helmet or any protective headgear. you should conclude 'Cannot determine'. "
        "If there is insufficient information to determine whether they were wearing a helmet, conclude 'Cannot determine'. "
        "Provide your answer as [Explanation here] 'Yes', 'No', or 'Cannot determine' and appropriate number 0 if the answer is no, 1 if the answer is yes and 2 if the answer is Cannot determine."
        "Ensure your explanation is concise and directly related to the information provided in the clinical note."
    )

    user_prompt_message = (
        f"You are a pragmatic and focused medical nurse. Please analyze the following clinical note and determine if the injured person was wearing a helmet or not or cannot be determined. "
        f"Provide your answer as [EXPLANATION AND THINKING] followed by 'Yes', 'No', or 'Cannot determine' finished your response with the number 0 if the answer is no, 1 if the answer is yes and 2 if the answer is Cannot determine. \n\n"
        f"For example: Based on the information provided, the patient was not wearing a helmet. as the note mentions that the injured has no head protection in the clinical note. No, 0 \n\n"
        f"Another example: Based on the information provided, the patient was wearing a helmet. as the note mentions that the injured was wearing a helm in the clinical note. Yes, 1 \n\n"
        f"Double check if there is a '-' minus or + plus sign before the helmet, it should be considered as no helmet or helmet respectively. \n\n"
        f"If you don't know the answer for sure or there are no relevant mentions or text, then you must say that you cannot determine the answer. Cannot determine, 2 \n\n"
        f"MAKE SURE TO END YOUR RESPONSE WITH THE NUMBER ONLY"
    )

    system_prompt_message = (
        "You are an expert in medical note analysis specializing in determining whether a patient was wearing a helmet during an accident. Your task is to analyze the data provided and determine if the patient was wearing a helmet, not wearing a helmet, or if it cannot be determined. Consider all relevant information in the data and only in the provided data. For example, if the note explicitly states that the patient was wearing a helmet, classify it as CLASS 0 (Helmet). If the note indicates that the patient was not wearing a helmet, classify it as CLASS 1 (No Helmet). If the note does not provide clear information about the helmet status, classify it as CLASS 2 (cannot determine). Provide your answer as Helmet, No Helmet, cannot determine and appropriate number 0 if the answer is Helmet, 1 if the answer is No Helmet and 2 if the answer is cannot determine."
    )

    user_prompt_message = (
        f"You are a medical note classifier. Please analyze the following free text data provided and determine if the patient was wearing a helmet, not wearing a helmet, or if it cannot be determined. Here are the notes: [Insert the unstructured notes here]."
    )

    asyncio.run(start_eval(system_prompt=system_prompt_message, user_prompt=user_prompt_message))

    evaluate_classification_accuracy('NEISS data/neiss_filtered_labeled_output.csv')