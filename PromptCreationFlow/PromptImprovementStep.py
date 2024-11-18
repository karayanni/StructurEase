import pandas as pd


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
    return response_dict


# TODO: Implement the logic to actually iterate over the labeled data and improve the prompt - Add any other input params as needed
def ImprovePrompt(current_prompt: dict, all_labeled_data: pd.DataFrame, misclassified: pd.DataFrame, classification_request: str, classes: list) -> dict:
    """
    This function is responsible for improving the prompt by iterating over the labeled data and improving the prompt
    It uses the miss-classified notes as few shots to improve the classification prompt.
    """
    # print(f"current_prompt: {current_prompt} - all_labeled_data: {all_labeled_data} - misclassified: {misclassified} - classification_request: {classification_request} - classes: {classes}")

    if len(misclassified) == 0:
        print("No misclassified data found. Prompt remains unchanged.")
        return current_prompt
    else:
        client = OpenAI()
        few_shot_examples = ""
        for index, row in misclassified.iterrows():
            few_shot_examples += f"\n Example: {row['Sampled Text']} -> Expected Label: {row['Label']} \n"

        user_message_content = (
            "Generate a new JSON object with two keys: 'system_message' and 'user_message' by improving the current one. RESPONSE WITH THE STRING THAT REPRESENT THE JSON.\n"
            "The 'system_message' should instruct the assistant on how to classify unstructured notes based on the following criteria:\n\n"
            f"Classification Request: {classification_request}\n\n"
            f"Classification Classes: {', '.join([f'CLASS {i} NAME: {cls}' for i, cls in enumerate(classes)])}\n\n"
            " \n You will be provided two main inputs. 1. The current JSON object of the prompts 2. The notes which it failed to classify. \n\n"
            "\n\n Your task is to improve the prompt according to the mistakes and add the mistakes as additional examples from the misclassified data so that the prompt will include them as examples on how to classify. \n\n"
            f"\n\n CURRENT PROMPT: {current_prompt} \n\n END OF CURRENT PROMPT \n\n"
            "YOU NEED TO IMPROVE THE ABOVE PROMPT BY ADDING FEW SHOT EXAMPLES FROM THE MISCLASSIFIED DATA.\n\n"
            " THINK STEP BY STEP WHY THE CURRENT PROMPT FAILED AND ADD INSTRUCTIONS TO IMPROVE THE PROMPT MAKING IT SMARTER AND PREVENT THE MISTAKE BY ADDING INSTRUCTIONS TO FOLLOW TO PREVENT THE MISTAKE.\n\n"
            "Here are a few examples of misclassified data that you need to add to the prompt as examples to help improving the prompts effectiveness.\n"
            f"{few_shot_examples}\n\n"
            "\n Use the style of the following example for structure, tone, and clarity. MAKE SURE YOUR OUTPUT IS A STRING THAT REFLECTS THE DICTIONARY STARTING WITH \" AND ENDING WITH \" - MAKE SURE YOUR OUTPUT IS A VALID JSON STRING. \n\n"
            "Here is an example to follow - MAKE SURE TO USE THE SAME STRUCTURE:\n\n"
            "{"
            "\"system_message\":\n"
            "\"You are an expert in [COMPLETE ACCORDING TO CONTEXT HERE] specializing in [COMPLETE ACCORDING TO CLASSIFICATION REQUEST HERE]. \n\n "
            "Your task is to analyze the data provided and determine [COMPLETE ACCORDING TO CONTEXT HERE CLASSIFICATION REQUEST AND CLASSIFICATION CLASSES] \n\n"
            "Consider all relevant information in the data and only in the provided data. [ADD ANY SPECIFIC EXAMPLE DETAILS AND INFO THAT CAN BE RELEVANT FROM THE EXAMPLES YOU SEE] "
            "Provide your answer as [Explanation here] [CLASS 0 NAME HERE], [CLASS 1 NAME HERE], ...[CLASS K NAME HERE] and appropriate number 0 if the answer is [CLASS 0], 1 if the answer is [CLASS 1 NAME HERE] and so on.\"\n\n"
            "\"user_message\":\n"
            "\"You are a [COMPLETE ACCORDING TO CONTEXT HERE]. Your task is to read an unstructured text and classify it according to the given classes. [APPEND FEW SHOT EXAMPLES HERE]. Please analyze the following free text data provided and determine [COMPLETE ACCORDING TO CLASSIFICATION CONTEXT HERE]. MAKE SURE TO END YOUR RESPONSE WITH THE NUMBER ONLY.\"\n\n"
            "}"
        )

        # System message
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Please provide a JSON response with two keys: 'system_message' and 'user_message'. "
                " YOUR TASK IS TO TAKE THE CURRENT JSON OF THE PROMPTS AND IMPROVE IT ACCORDING TO THE GIVEN EXAMPLES WHERE THE CURRENT PROMPT FAILED."
                " Each of the prompts should follow the style and purpose demonstrated in the examples, tailoring instructions to classify unstructured notes as described."
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

        response_dict["system_message"] = response_dict[
                                              "system_message"] + "\n\n AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1"
        response_dict["user_message"] = response_dict[
                                            "user_message"] + "\n\n AGAIN MAKE SURE YOUR ANSWER ENDS WITH A DIGIT ONLY FOR EXAMPLE 1"

    return response_dict


if __name__ == '__main__':
    p_0 = {'system_message': "You are an expert in medical note analysis specializing in determining whether a patient was wearing a helmet during an accident. Your task is to analyze the data provided and determine if the patient was wearing a helmet, not wearing a helmet, or if it cannot be determined based on the unstructured notes. Consider all relevant information in the data and only in the provided data. For example, if the note explicitly states that the patient was wearing a helmet, classify it as 'Helmet'. If the note indicates that the patient was not wearing a helmet or does not mention a helmet at all, classify it as 'No Helmet'. If the note is ambiguous or lacks sufficient information to make a determination, classify it as 'cannot determine'. Provide your answer as Helmet, No Helmet, and cannot determine and appropriate number 0 if the answer is Helmet, 1 if the answer is No Helmet and 2 if the answer is cannot determine.", 'user_message': 'You are a medical note classifier. Please analyze the following free text data provided and determine if the patient was wearing a helmet, not wearing a helmet, or if it cannot be determined based on the unstructured notes. 15 YOM RIDING AN ELECTRIC SCOOTER AND FELL OFF. DX KNEE ABRASION, SHOULDER SPRAIN. ^29YOM C/O LEFT WRIST PAIN AND HEADACHE S/P FALLING OFF BIKE. PT WAS RIDING BIKE WHEN SOMEONE OPENED A CAR DOOR AND HIT HIM. DX: FALL, WRIST PAIN, HEADACHE. 28 YOM WAS RIDING A BIKE WHEN SOMEONE TRUNED IN FRON OF HIM, SLAMMED ON HIS BRAKE AND WENT FORWARD OVER THE HANDLEBARS. WAS WEARING A HELMET. PAIN & ABRASION TO LT SHOULDER, LT FOREARM AND LT KNEE. DX: ACROMIOCLAVICULAR JOINT SEPARATION, TYPE 3; ABRASIONS. 64YOM WAS RIDING HIS BICYCLE WHEN HE WAS STRUCK BY A VEHICLE AND FLEW 15FT. ABRASION OVER RIGHT HAND AND BILATERAL KNEES NOTED ON EXAM. DX: FINGER PAIN,RIGHT, BIKE ACCIDENT. 21 YOM REPORTS RIDING AN ELECTRIC SCOOTER AND FELL, HAS ELBOW PAIN. DX: NONE, LEFT WITHOUT TREATMENT COMPLETE. 21YOF PRESENTS AFTER BEING STRUCK BY A CAR WHILE RIDING HER BIKE IN THE STREET. REPORTS BEING HIT ON HER RIGHT SIDE. DX: ABRASION TO KNEE. 11 YOM RIDING A BIKE AND FELL OFF, HIT HEAD. DX CONCUSSION. 12 YOM TRYING TO DO A TRICK ON A BIKE AND FELL OFF. DX HAND CONTUSION. 62YOM, C/O LACERATION TOP OF HEAD AFT FALLING OFF BICYCLE DX: SCALP LACERATION, BILATERAL SHOULDER CONTUSION. 10 YOM RIDING A BIKE, HIT A BUMP AND FELL OFF. DX TIBIA FX. 0'}

    labeled_data = {
        "Sampled Text": [
            "53 YOM FELL RIDING BICYCLE  DX;  R KNEE LAC, F...",
            "23YOM FELL OFF OF BICYCLE, LANDING ON RT SHOUL...",
            "39YOM PRESENTED AFTER BEING HIT BY A CAR WHILE...",
            "8 YOM RIDING A BIKE AND HIT BY A CAR.  DX HAND...",
            "32YOM PRESENTED TO ED C/O FALL, PT STATED A CA...",
            "13YOF PATIENT WAS ON A GAS POWERED BICYCLE THA...",
            "42YOM PRESENTS WITH HAND LACERATION AFTER BICY...",
            "6YOF- PT WAS RIDING ON HER BIKE WHEN SHE FELL ...",
            "46YOF PRESENT TO ED W/ L ELBOW/WRIST/HAND PX A...",
            "18 MOF CLIMBING OVER A BIKE AND FELL, HIT FACE...",
            "31YOM PRESENTED AT ED C/O OF FALL. PT FELL OFF...",
            "5 YOM INJURED FOOT, DAD'S MOPED FELL OVER ONTO...",
            "31YOM PATIENT FELL OFF A MOTORZIED SCOOTER DOW...",
            "16YOM PRESENTS WITH ELBOW PAIN AFTER HE WAS RI...",
            "32YOF PATIENT FELL ON AN OUTSTRETCHED HAND WHI...",
            "65YOF REPORTS SHE WAS RIDING A BIKE YESTERDAY ...",
            "46YOM RIDING MOPED, HIT POTHOLE AND FELL OFF A...",
            "17YOF WAS RIDING HER BICYCLE DOWN THE SIDEWALK...",
            "64YOF STS FELL FROM SCOOTER 2 DAYS AGO, ELECTR...",
            "7YOM WAS RIDING A SCOOTER AND FELL FORWARD W/ ...",
            "29 YOM HIT A POTHOLE WHILE RIDING MOTORIZED SC...",
            "35 YOM HAD A MOUNTAIN BIKING ACCIDENT 4 DAYS A...",
            "10YOM FELL OFF BIKE HELEMETED DX RENAL HEMATOMA",
            "24YOM WAS RIDING A BIKE HELMETED OVER SEVERAL ...",
            "31YOF CRASHED ON HER BICYCLE YESTERDAY WHILE R...",
            "5YOM RIDING BIKE IN STREET WHILE WEARING HELME...",
            "56YOM PW FALLING OFF OF MOTORIZED SCOOTER, PT ...",
            "65YOM WAS RIDING HIS BICYCLE WITH A HELMET WHE...",
        ],
        "Label": [
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "cannot determine",
            "Helmet",
            "Helmet",
            "Helmet",
            "Helmet",
            "Helmet",
            "Helmet",
            "Helmet",
            "Helmet",
        ],
    }

    all_labeled_data = pd.DataFrame(labeled_data)

    data = {
        "Sampled Text": [
            "23YOM FELL OFF OF BICYCLE, LANDING ON RT SHOUL...",
            "32YOM PRESENTED TO ED C/O FALL, PT STATED A CA...",
            "13YOF PATIENT WAS ON A GAS POWERED BICYCLE THA...",
            "42YOM PRESENTS WITH HAND LACERATION AFTER BICY...",
            "6YOF- PT WAS RIDING ON HER BIKE WHEN SHE FELL ...",
            "46YOF PRESENT TO ED W/ L ELBOW/WRIST/HAND PX A...",
            "18 MOF CLIMBING OVER A BIKE AND FELL, HIT FACE...",
            "31YOM PRESENTED AT ED C/O OF FALL. PT FELL OFF...",
            "5 YOM INJURED FOOT, DAD'S MOPED FELL OVER ONTO...",
            "31YOM PATIENT FELL OFF A MOTORZIED SCOOTER DOW...",
            "17YOF WAS RIDING HER BICYCLE DOWN THE SIDEWALK...",
        ],
        "Label": ["cannot determine"] * 11,
        "LLM_number": [1] * 11,
    }

    miss_classified = pd.DataFrame(data)
    classification_request = "determine whether a patient was wearing a helmet during an accident"
    classes = ["Helmet", "No Helmet", "cannot determine"]
    test_run = ImprovePrompt(p_0, all_labeled_data, miss_classified, classification_request, classes)
