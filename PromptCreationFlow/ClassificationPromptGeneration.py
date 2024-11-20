import json
import random
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


def exponential_backoff_retry(func, max_retries=4, initial_delay=0.5, max_delay=16, jitter=0.5):
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
            time.sleep(delay + random.uniform(0, jitter))
            delay = min(max_delay, delay * 2)
            logging.error(f"Retrying after {delay} seconds... (Attempt {retries}/{max_retries})")


def InitialGenerateClassificationPrompt(df, column_name, classification_request, classes):
    client = OpenAI()

    sample_notes = df[column_name].sample(10, random_state=42).tolist()

    # TODO: Improve the prompt by adding think step by step and adding the explanation before the answer.
    user_message_content = (
        "Generate a JSON object with two keys: 'system_message' and 'user_message'. RESPONSE WITH THE STRING THAT REPRESENT THE JSON.\n"
        "The 'system_message' should instruct the assistant on how to classify unstructured notes based on the following criteria:\n\n"
        # f"In the generated prompt, ask in the 'user_message' to think step by step and provide a 1 sentence reasoning before providing the final answer. \n\n"
        f"Classification Request: {classification_request}\n\n"
        f"Classification Classes: {', '.join([f'CLASS {i} NAME: {cls}' for i, cls in enumerate(classes)])}\n\n"
        "Provided below are 10 sample inputs that reflect what the prompt should expect. you can use these examples to better craft the prompt.\n\n"
        )
    for idx, note in enumerate(sample_notes, start=1):
        user_message_content += f"\nExample {idx}: \"{note}\" \n"

    user_message_content_continuation = (
        "\n Use the style of the following example for structure, tone, and clarity. MAKE SURE YOUR OUTPUT IS A STRING THAT REFLECTS THE DICTIONARY STARTING WITH \" AND ENDING WITH \" - MAKE SURE YOUR OUTPUT IS A VALID JSON STRING. \n\n"
        "Here is an example to follow - MAKE SURE TO USE THE SAME STRUCTURE:\n\n"
        "{"
        "\"system_message\":\n"
        "\"You are an expert in [COMPLETE ACCORDING TO CONTEXT HERE] specializing in [COMPLETE ACCORDING TO CLASSIFICATION REQUEST HERE]. \n\n "
        "Your task is to analyze the data provided and determine [COMPLETE ACCORDING TO CONTEXT HERE CLASSIFICATION REQUEST AND CLASSIFICATION CLASSES] \n\n"
        "Consider all relevant information in the data and only in the provided data. [ADD ANY SPECIFIC EXAMPLE DETAILS AND INFO THAT CAN BE RELEVANT FROM THE EXAMPLES YOU SEE] "
        "Provide your answer as [Explanation here] [CLASS 0 NAME HERE], [CLASS 1 NAME HERE], ...[CLASS K NAME HERE] and appropriate number 0 if the answer is [CLASS 0], 1 if the answer is [CLASS 1 NAME HERE] and so on.\"\n\n"
        "\"user_message\":\n"
        "\"You are a [COMPLETE ACCORDING TO CONTEXT HERE]. Your task is to read an unstructured text and classify it according to the given classes. Please analyze the following free text data provided and determine [COMPLETE ACCORDING TO CLASSIFICATION CONTEXT HERE]. MAKE SURE TO END YOUR RESPONSE WITH THE NUMBER ONLY.\"\n\n"
        # "\"You are a [COMPLETE ACCORDING TO CONTEXT HERE]. Your task is to read an unstructured text and classify it according to the given classes. Think step by step and analyze the following free text data provided and determine [COMPLETE ACCORDING TO CLASSIFICATION CONTEXT HERE]. Provide your answer as [EXPLANATION AND REASONING], [CLASSIFICATION] [CLASSIFICATION NUMBER]  MAKE SURE TO END YOUR RESPONSE WITH THE NUMBER ONLY.\"\n\n"
        "}"
    )

    user_message_content += user_message_content_continuation

    # System message
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Please provide a JSON response with two keys: 'system_message' and 'user_message'. "
            "Each should follow the style and purpose demonstrated in the examples, tailoring instructions to classify unstructured notes as described."
        )
    }

    # User message
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    message_list = [system_message, user_message]

    # Function to make the API call
    def make_completion_request():
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            top_p=1,
            messages=message_list
        )
        return completion

    # Use exponential backoff to retry the API call in case of rate limits or transient errors
    completion = exponential_backoff_retry(make_completion_request)
    assistant_response = completion.choices[0].message.content.strip()

    response_dict = json.loads(assistant_response)

    response_dict["system_message"] = response_dict["system_message"] + "\n\n AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1"
    response_dict["user_message"] = response_dict["user_message"] + "\n\n AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1"

    # response_dict["user_message"] = response_dict["user_message"] + "\n\n AGAIN MAKE SURE YOUR RESPONSE IS IN THIS FORM [1 SENTENCE REASONING],[CLASSIFICATION CLASS],[CLASSIFICATION NUMBER] " \
    #                                                                 "\n Example Answer: [...step by step reasoning here ...], class 1,1" \
    #                                                                 "\n Example Answer: [...step by step reasoning here ...], class 3,3" \                                                                 "\n\n  AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1"
    return response_dict
