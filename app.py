import streamlit as st
import pandas as pd
from PromptCreationFlow.ClassificationPromptGeneration import GenerateClassificationPrompt


def main():
    st.title("Classification Request App")

    # Step 1: Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        if len(df) < 1000:
            st.warning("The CSV file must contain at least 1000 rows for StructurEase.")
            return

        st.write("Preview of CSV file:")
        st.write(df.head())

        # Step 2: Select a column
        column_name = st.selectbox("Select the column for classification", df.columns)

        # Step 3: Enter classification request
        classification_request = st.text_area("Enter the classification request")

        # Step 4: Enter qualification classes (at least 2 options)
        qualification_classes = st.text_input("Enter qualification classes (comma-separated)")
        classes = [item.strip() for item in qualification_classes.split(',') if item.strip()]

        if len(classes) >= 2:
            if len(classes) > 9:
                st.warning("Please add at most 9 classes.")
            else:
                st.subheader("Manual Labeling for Sampled Data")

                # Sample K random rows from the selected column
                # todo: instead of sampling 10 rows randomly, run a first pass of the model to get the most 'uncertain'ish rows
                # todo: make this sampling a function and give it a good name to explain in the paper.
                sampled_data = df[column_name].sample(10, random_state=42).reset_index(drop=True)
                labels = []

                # Create a form to label the sampled rows
                with st.form("labeling_form"):
                    st.write("Please label each row below:")

                    for i, row in enumerate(sampled_data):
                        label = st.selectbox(f"Row {i + 1}: {row}", options=classes, key=f"label_{i}")
                        labels.append(label)

                    submit_labels = st.form_submit_button("Submit Labels")

                if submit_labels:
                    labeled_data = pd.DataFrame({"Sampled Text": sampled_data, "Label": labels})
                    st.write("Labeled Data Preview:")
                    st.write(labeled_data)
                # Step 5: Submit button
                if st.button("Submit"):
                    # Call a function with these parameters
                    st.write("CSV Column Selected:", column_name)
                    st.write("Classification Request:", classification_request)
                    st.write("Qualification Classes:", classes)

                    # Placeholder for your function
                    st.write(GenerateClassificationPrompt(df, column_name, classification_request, classes))
        else:
            st.warning("Please enter at least two qualification classes.")


if __name__ == "__main__":
    main()
