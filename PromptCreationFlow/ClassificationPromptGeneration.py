import pandas as pd
import asyncio
import random
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


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


def GenerateClassificationPrompt(df, column_name, classification_request, classes):
    client = OpenAI()

    # Split the data into 80% validation and 20% test set
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    sample_notes = train_df[column_name].sample(10, random_state=42).tolist()

    # Construct message to request a classification prompt generation
    user_message_content = (
        "Generate a JSON object with two keys: 'system_message' and 'user_message'.\n"
        "The 'system_message' should instruct the assistant on how to classify unstructured notes based on the following criteria:\n\n"
        f"Classification Request: {classification_request}\n\n"
        f"Classification Classes: {', '.join([f'CLASS {i} NAME: {cls}' for i, cls in enumerate(classes)])}\n\n"
        "Use the style of the following example for structure, tone, and clarity:\n\n"
        "System example:\n"
        "\"You are an expert in [COMPLETE ACCORDING TO CONTEXT HERE] specializing in [COMPLETE ACCORDING TO CLASSIFICATION REQUEST HERE]. \n\n "
        "Your task is to analyze the data provided and determine [COMPLETE ACCORDING TO CONTEXT HERE CLASSIFICATION REQUEST AND CLASSIFICATION CLASSES] \n\n"
        "Consider all relevant information in the data and only in the provided data. [ADD ANY SPECIFIC EXAMPLE DETAILS AND INFO THAT CAN BE RELEVANT FROM THE EXAMPLES YOU SEE] "
        "Provide your answer as [Explanation here] [CLASS 0 NAME HERE], [CLASS 1 NAME HERE], ...[CLASS K NAME HERE] and appropriate number 0 if the answer is [CLASS 0], 1 if the answer is [CLASS 1 NAME HERE] and so on.\"\n\n"
        "User example:\n"
        "\"You are a [COMPLETE ACCORDING TO CONTEXT HERE]. Please analyze the following free text data provided and determine [COMPLETE ACCORDING TO CLASSIFICATION CONTEXT HERE]\"\n\n"
        "MAKE SURE TO END YOUR RESPONSE WITH THE NUMBER ONLY. \n\n"
        "Provided below are 10 sample unstructured notes to use in the prompt generation for enriching the prompt and providing examples inside the prompt: \n\n"
    )

    for idx, note in enumerate(sample_notes, start=1):
        user_message_content += f"\nExample {idx}: \"{note}\" \n"

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

    return assistant_response
