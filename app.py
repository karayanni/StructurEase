import streamlit as st
import pandas as pd

from PromptCreationFlow.ChooseLabelingData import ChooseLabelingData
from PromptCreationFlow.ClassificationPromptGeneration import InitialGenerateClassificationPrompt

#
# def labeling_form(df, column_name, classes):
#     # Create a form to label the sampled rows
#     for i in range(st.session_state.current_iteration+1):
#         with st.form(f"labeling_form_{i}"):
#
#             # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
#             sampled_indices, sampled_values = ChooseLabelingData(df, column_name, st.session_state.current_prompt,
#                                                                  st.session_state.already_chosen_data_indices)
#
#             st.session_state.already_chosen_data_indices.extend(sampled_indices)
#
#             st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")
#
#             st.subheader(f"Manual Labeling Iteration: {st.session_state.current_iteration}")
#             labels = []
#             for i, row in enumerate(sampled_values):
#                 label = st.selectbox(f"Row {i + 1}: {row}", options=classes, key=f"label_{i}")
#                 labels.append(label)
#
#             submit_labels = st.form_submit_button("Submit Labels")
#
#             if submit_labels:
#                 labeled_data = pd.DataFrame({"Sampled Text": sampled_values, "Label": labels})
#                 st.session_state.all_labeled_data = pd.concat([st.session_state.all_labeled_data, labeled_data],
#                                                               ignore_index=True)
#
#                 st.write("Labeled Data Preview:")
#                 st.write(st.session_state.all_labeled_data)
#
#                 # STEP 3 - Return Evaluation of the Labeled Data to User
#                 st.write(f"Evaluation of Prompt_{i} on the manually Labeled Data")
#                 # todo: add here the evaluation function call...
#
#                 # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
#                 # todo: add here the prompt update function call...
#
#                 # Step 5: Starting the Final Classification on the entire dataset
#
#     if st.button(f"Another Labeling Iteration - Iteration {st.session_state.current_iteration}"):
#         st.session_state.current_iteration += 1
#         labeling_form(df, column_name, classes)


def main():
    st.title("Classification Request App")

    # Step 1: Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Input 1: upload CSV file
        df = pd.read_csv(uploaded_file)

        if len(df) < 1000:
            st.warning("The CSV file must contain at least 1000 rows for StructurEase.")
            return

        st.write("Preview of CSV file:")
        st.write(df.head())

        # Input 2: Select a column
        column_name = st.selectbox("Select the column for classification", df.columns)

        # Input 3: Enter classification request
        classification_request = st.text_area("Enter the classification request")

        # Input 4: Enter qualification classes (at least 2 options)
        qualification_classes = st.text_input("Enter qualification classes (comma-separated)")
        classes = [item.strip() for item in qualification_classes.split(',') if item.strip()]

        if len(classes) >= 2:
            if len(classes) > 9:
                st.warning("Please add at most 9 classes.")
            else:
                # STEP 1 - CREATE PROMPT_0
                if "current_prompt" not in st.session_state:
                    prompt_0 = InitialGenerateClassificationPrompt(df, column_name, classification_request, classes)
                    st.session_state.current_prompt = prompt_0
                if "current_iteration" not in st.session_state:
                    st.session_state.current_iteration = 0

                if "already_chosen_data_indices" not in st.session_state:
                    st.session_state.all_labeled_data = pd.DataFrame(columns=["Sampled Text", "Label"])
                    st.session_state.already_chosen_data_indices = []

                # Create a form to label the sampled rows
                with st.form("labeling_form_0"):
                    # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                    sampled_indices, sampled_values = ChooseLabelingData(df, column_name, st.session_state.current_prompt,
                                                                         st.session_state.already_chosen_data_indices)

                    st.session_state.already_chosen_data_indices.extend(sampled_indices)

                    st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                    st.subheader(f"Manual Labeling Iteration: 0")
                    labels = []
                    for i, row in enumerate(sampled_values):
                        label = st.selectbox(f"Row {i + 1}: {row}", options=classes, key=f"label_{st.session_state.current_iteration}_{i}")
                        labels.append(label)

                    submit_labels = st.form_submit_button("Submit Labels 0")

                    if submit_labels:
                        labeled_data = pd.DataFrame({"Sampled Text": sampled_values, "Label": labels})
                        st.session_state.all_labeled_data = pd.concat([st.session_state.all_labeled_data, labeled_data], ignore_index=True)

                        st.write("Labeled Data Preview:")
                        st.write(st.session_state.all_labeled_data)

                        # STEP 3 - Return Evaluation of the Labeled Data to User
                        st.write(f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")
                        # todo: add here the evaluation function call...

                        # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                        # todo: add here the prompt update function call...

                        # Step 5: Starting the Final Classification on the entire dataset

                if st.button("Add Labeling Iteration 1"):
                    with st.form("labeling_form_1"):
                        # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                        sampled_indices, sampled_values_1 = ChooseLabelingData(df, column_name,
                                                                             st.session_state.current_prompt,
                                                                             st.session_state.already_chosen_data_indices)

                        st.session_state.already_chosen_data_indices.extend(sampled_indices)

                        st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                        st.subheader(f"Manual Labeling Iteration: 1")
                        labels = []
                        for i, row in enumerate(sampled_values_1):
                            label = st.selectbox(f"Row {i + 1}: {row}", options=classes, key=f"label_1_{i}")
                            labels.append(label)

                        submit_labels = st.form_submit_button("Submit Labels 1")

                        if submit_labels:
                            labeled_data = pd.DataFrame({"Sampled Text": sampled_values, "Label": labels})
                            st.session_state.all_labeled_data = pd.concat(
                                [st.session_state.all_labeled_data, labeled_data], ignore_index=True)

                            st.write("Labeled Data Preview:")
                            st.write(st.session_state.all_labeled_data)

                            # STEP 3 - Return Evaluation of the Labeled Data to User
                            st.write(
                                f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")
                            # todo: add here the evaluation function call...

                            # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                            # todo: add here the prompt update function call...

                            # Step 5: Starting the Final Classification on the entire dataset

                    if st.button("Add Labeling Iteration 2"):
                        with st.form("labeling_form_2"):
                            # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                            sampled_indices, sampled_values_2 = ChooseLabelingData(df, column_name,
                                                                                 st.session_state.current_prompt,
                                                                                 st.session_state.already_chosen_data_indices)

                            st.session_state.already_chosen_data_indices.extend(sampled_indices)

                            st.write(
                                f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                            st.subheader(f"Manual Labeling Iteration: {st.session_state.current_iteration}")
                            labels = []
                            for i, row in enumerate(sampled_values_2):
                                label = st.selectbox(f"Row {i + 1}: {row}", options=classes, key=f"label_2_{i}")
                                labels.append(label)

                            submit_labels = st.form_submit_button("Submit Labels 2")

                            if submit_labels:
                                labeled_data = pd.DataFrame({"Sampled Text": sampled_values, "Label": labels})
                                st.session_state.all_labeled_data = pd.concat(
                                    [st.session_state.all_labeled_data, labeled_data], ignore_index=True)

                                st.write("Labeled Data Preview:")
                                st.write(st.session_state.all_labeled_data)

                                # STEP 3 - Return Evaluation of the Labeled Data to User
                                st.write(
                                    f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")
                                # todo: add here the evaluation function call...

                                # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                                # todo: add here the prompt update function call...

                                # Step 5: Starting the Final Classification on the entire dataset

                if st.button("Start Final Classification Using Latest Prompt"):
                    # Call a function with these parameters
                    st.write(f"Final Classification Prompt: {st.session_state.current_prompt}")

                    # todo: add here the final classification function call...
                    st.write("Final Classification Started")

        else:
            st.warning("Please enter at least two qualification classes.")


if __name__ == "__main__":
    main()
