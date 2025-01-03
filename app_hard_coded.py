import streamlit as st
import pandas as pd

from Evaluation.EvaluateCurrentPromptOnEntireDS import evaluate_classification_accuracy_on_entire_DS, run_classification_on_entire_DS
from PromptCreationFlow.ChooseLabelingData import ChooseLabelingData, ChooseLabelingDataRandom
from PromptCreationFlow.ClassificationPromptGeneration import InitialGenerateClassificationPrompt
from PromptCreationFlow.EvaluateCurrentPrompt import EvaluateCurrentPrompt
from PromptCreationFlow.PromptImprovementStep import ImprovePrompt


def main():
    st.title("Classification Request App")

    # Step 1: Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Input 1: upload CSV file
        df = pd.read_csv(uploaded_file)

        if len(df) < 400:
            st.warning("The CSV file must contain at least 400 rows for StructurEase.")
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

                with st.form(f"labeling_form_0"):
                    # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                    if "sample_0" not in st.session_state:
                        # evaluate_classification_accuracy_on_entire_DS(st.session_state.current_prompt["system_message"], st.session_state.current_prompt["user_message"], "P_0_e2e")
                        sampled_indices, sampled_values = ChooseLabelingData(df, column_name, st.session_state.current_prompt,
                                                                         st.session_state.already_chosen_data_indices)
                        st.session_state.sample_0 = sampled_values

                        st.session_state.already_chosen_data_indices.extend(sampled_indices)

                    st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                    st.subheader(f"Manual Labeling Iteration: {st.session_state.current_iteration}")
                    labels = []
                    for i, row in enumerate(st.session_state.sample_0):
                        label = st.selectbox(f"Row {i + 1}: {row}", options=classes, key=f"label_0_{i}")
                        labels.append(label)

                    submit_labels = st.form_submit_button("Submit")

                    if submit_labels:
                        newly_labeled_data = pd.DataFrame({"Sampled Text": st.session_state.sample_0, "Label": labels})
                        combined_labeled_data = pd.concat([st.session_state.all_labeled_data, newly_labeled_data], ignore_index=True)
                        st.write("Labeled Data Preview:")
                        st.write(combined_labeled_data)
                        st.session_state.all_labeled_data = combined_labeled_data

                        st.write(
                            f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")

                        # STEP 3 - Return Evaluation of the Labeled Data to User
                        eval_report, misclassified = EvaluateCurrentPrompt(st.session_state.current_prompt["system_message"],
                                              st.session_state.current_prompt["user_message"],
                                              st.session_state.all_labeled_data,
                                              classes)

                        st.write(eval_report)
                        st.write("Misclassified notes: ")
                        st.write(misclassified)

                        # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                        st.session_state.current_prompt = ImprovePrompt(st.session_state.current_prompt, st.session_state.all_labeled_data, misclassified, classification_request, classes)

                        st.session_state.current_iteration += 1

                if st.session_state.current_iteration > 0:
                    with st.form(f"labeling_form_1"):
                        # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                        if "sample_1" not in st.session_state:
                            # evaluate_classification_accuracy_on_entire_DS(
                            #     st.session_state.current_prompt["system_message"],
                            #     st.session_state.current_prompt["user_message"], "P_1_random_6")

                            sampled_indices, sampled_values = ChooseLabelingData(df, column_name,
                                                                             st.session_state.current_prompt,
                                                                             st.session_state.already_chosen_data_indices)
                            st.session_state.sample_1 = sampled_values

                            st.session_state.already_chosen_data_indices.extend(sampled_indices)

                        st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                        st.subheader(f"Manual Labeling Iteration: {st.session_state.current_iteration}")
                        labels = []
                        for i, row in enumerate(st.session_state.sample_1):
                            label = st.selectbox(f"Row {i + 1}: {row}", options=classes,
                                                 key=f"label_1_{i}")
                            labels.append(label)

                        submit_labels = st.form_submit_button("Submit")

                        if submit_labels:
                            newly_labeled_data = pd.DataFrame({"Sampled Text": st.session_state.sample_1, "Label": labels})
                            combined_labeled_data = pd.concat([st.session_state.all_labeled_data, newly_labeled_data],
                                                              ignore_index=True)
                            st.write("Labeled Data Preview:")
                            st.write(combined_labeled_data)
                            st.session_state.all_labeled_data = combined_labeled_data

                            st.write(
                                f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")

                            # STEP 3 - Return Evaluation of the Labeled Data to User
                            eval_report, misclassified = EvaluateCurrentPrompt(
                                st.session_state.current_prompt["system_message"],
                                st.session_state.current_prompt["user_message"],
                                st.session_state.all_labeled_data,
                                classes)

                            st.write(eval_report)
                            st.write("Misclassified notes: ")
                            st.write(misclassified)
                            # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                            st.session_state.current_prompt = ImprovePrompt(st.session_state.current_prompt,
                                                                            st.session_state.all_labeled_data,
                                                                            misclassified, classification_request,
                                                                            classes)

                            st.session_state.current_iteration += 1
                # Hardcoded up to 2 iterations of manual labeling.
                if st.session_state.current_iteration > 1:
                    with st.form(f"labeling_form_2"):
                        # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                        if "sample_2" not in st.session_state:

                            # evaluate_classification_accuracy_on_entire_DS(
                            #     st.session_state.current_prompt["system_message"],
                            #     st.session_state.current_prompt["user_message"], "P_2_7")

                            sampled_indices, sampled_values = ChooseLabelingData(df, column_name,
                                                                             st.session_state.current_prompt,
                                                                             st.session_state.already_chosen_data_indices)
                            st.session_state.sample_2 = sampled_values

                            st.session_state.already_chosen_data_indices.extend(sampled_indices)

                        st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                        st.subheader(f"Manual Labeling Iteration: {st.session_state.current_iteration}")
                        labels = []
                        for i, row in enumerate(st.session_state.sample_2):
                            label = st.selectbox(f"Row {i + 1}: {row}", options=classes,
                                                 key=f"label_2_{i}")
                            labels.append(label)

                        submit_labels = st.form_submit_button("Submit")

                        if submit_labels:
                            newly_labeled_data = pd.DataFrame({"Sampled Text": st.session_state.sample_2, "Label": labels})
                            combined_labeled_data = pd.concat([st.session_state.all_labeled_data, newly_labeled_data],
                                                              ignore_index=True)
                            st.write("Labeled Data Preview:")
                            st.write(combined_labeled_data)
                            st.session_state.all_labeled_data = combined_labeled_data

                            st.write(
                                f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")

                            # STEP 3 - Return Evaluation of the Labeled Data to User
                            eval_report, misclassified = EvaluateCurrentPrompt(
                                st.session_state.current_prompt["system_message"],
                                st.session_state.current_prompt["user_message"],
                                st.session_state.all_labeled_data,
                                classes)

                            st.write(eval_report)
                            st.write("Misclassified notes: ")
                            st.write(misclassified)
                            # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                            st.session_state.current_prompt = ImprovePrompt(st.session_state.current_prompt,
                                                                            st.session_state.all_labeled_data,
                                                                            misclassified, classification_request,
                                                                            classes)

                            st.session_state.current_iteration += 1

                if st.session_state.current_iteration > 2:
                    with st.form(f"labeling_form_3"):
                        # STEP 2 - Use Current Prompt to Choose Data for Manual Labeling
                        if "sample_3" not in st.session_state:

                            # evaluate_classification_accuracy_on_entire_DS(
                            #     st.session_state.current_prompt["system_message"],
                            #     st.session_state.current_prompt["user_message"], "P_3_e2e")

                            sampled_indices, sampled_values = ChooseLabelingData(df, column_name,
                                                                             st.session_state.current_prompt,
                                                                             st.session_state.already_chosen_data_indices)
                            st.session_state.sample_3 = sampled_values

                            st.session_state.already_chosen_data_indices.extend(sampled_indices)

                        st.write(f"Prompt_{st.session_state.current_iteration}: \n\n {st.session_state.current_prompt}")

                        st.subheader(f"Manual Labeling Iteration: {st.session_state.current_iteration}")
                        labels = []
                        for i, row in enumerate(st.session_state.sample_3):
                            label = st.selectbox(f"Row {i + 1}: {row}", options=classes,
                                                 key=f"label_3_{i}")
                            labels.append(label)

                        submit_labels = st.form_submit_button("Submit")

                        if submit_labels:
                            newly_labeled_data = pd.DataFrame({"Sampled Text": st.session_state.sample_3, "Label": labels})
                            combined_labeled_data = pd.concat([st.session_state.all_labeled_data, newly_labeled_data],
                                                              ignore_index=True)
                            st.write("Labeled Data Preview:")
                            st.write(combined_labeled_data)
                            st.session_state.all_labeled_data = combined_labeled_data

                            st.write(
                                f"Evaluation of Prompt_{st.session_state.current_iteration} on the manually Labeled Data")

                            # STEP 3 - Return Evaluation of the Labeled Data to User
                            eval_report, misclassified = EvaluateCurrentPrompt(
                                st.session_state.current_prompt["system_message"],
                                st.session_state.current_prompt["user_message"],
                                st.session_state.all_labeled_data,
                                classes)

                            st.write(eval_report)
                            st.write("Misclassified notes: ")
                            st.write(misclassified)
                            # STEP 4 - Update the Prompt with the Labeled Data as Few-Shot Learning
                            st.session_state.current_prompt = ImprovePrompt(st.session_state.current_prompt,
                                                                            st.session_state.all_labeled_data,
                                                                            misclassified, classification_request,
                                                                            classes)

                            st.session_state.current_iteration += 1

                # Step 5: Starting the Final Classification on the entire dataset
                if st.button("Start Final Classification Using Latest Prompt"):
                    # Call a function with these parameters
                    st.write(f"Final Classification Prompt: {st.session_state.current_prompt}")

                    df_with_eval = run_classification_on_entire_DS(df, column_name,
                        st.session_state.current_prompt["system_message"],
                        st.session_state.current_prompt["user_message"], "final_results")

                    st.session_state.df_with_eval = df_with_eval
                    # preview the df_with_eval and add a button to download the entire CSV file.

                if "df_with_eval" in st.session_state:
                    st.write("Final Classification Results:")

                    st.write(st.session_state.df_with_eval.head())

                    st.download_button(
                        label="Download Classification Results",
                        data=st.session_state.df_with_eval.to_csv(index=False),
                        # Call the method and generate CSV content
                        file_name="final_classification_results.csv",
                        mime="text/csv"  # Set the MIME type
                    )
                    # st.download_button("Download Final Classification Results", st.session_state.df_with_eval.to_csv, "final_classification_results.csv")

        else:
            st.warning("Please enter at least two qualification classes.")


if __name__ == "__main__":
    main()
