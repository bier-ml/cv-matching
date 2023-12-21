import streamlit as st

from web import DOCX_TYPE
from web.process_data import read_docx


def main():
    col1, col2 = st.columns(2)
    with col1:
        col1.subheader("Upload your Document")
        uploaded_file = col1.file_uploader("Choose a file")
    with col2:
        col2.subheader("Document Content")

    if uploaded_file is not None:
        if uploaded_file.type == DOCX_TYPE:
            text = read_docx(uploaded_file)
            col2.info(text)
        else:
            st.warning("Please upload a docx file")
    else:
        col2.info("No document uploaded yet.")

    st.subheader("Interesting Job")
    job_name = st.text_input("Enter the job name")
    job_description = st.text_input("Enter the job description if applicable")


if __name__ == "__main__":
    # Streamlit configuration
    st.title("Document Reader")
    main()
