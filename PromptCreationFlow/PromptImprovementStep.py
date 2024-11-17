import pandas as pd


# TODO: Implement the logic to actually iterate over the labeled data and improve the prompt - Add any other input params as needed
def ImprovePrompt(current_prompt: dict, all_labeled_data: pd.DataFrame, misclassified: pd.DataFrame, classification_request: str, classes: list) -> dict:
    """
    This function is responsible for improving the prompt by iterating over the labeled data and improving the prompt
    It uses the miss-classified notes as few shots to improve the classification prompt.
    """
    print(f"current_prompt: {current_prompt} - all_labeled_data: {all_labeled_data} - misclassified: {misclassified} - classification_request: {classification_request} - classes: {classes}")
    # For now, we will just return the current prompt
    return current_prompt


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
