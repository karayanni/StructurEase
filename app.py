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
