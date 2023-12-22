import streamlit as st

from web import DOCX_TYPE
from web.process_data import read_docx, get_relevant_vacancies, check_relevance


def main():
    text, relevance = None, None
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

    st.subheader("Relevant Vacancies")
    if text:
        vacancies = get_relevant_vacancies(text)
        for i, vac in enumerate(vacancies):
            st.info(f"{i + 1}. {vac}")

    st.subheader("Any other interesting Vacancy?")
    job_name = st.text_input("Enter the job name")
    job_description = st.text_input("Enter the job description if applicable")

    if job_name:
        relevance = check_relevance(text, job_name, job_description)

    if relevance is not None:
        if relevance >= 0.75:
            st.success("This job is a good fit for your CV!")
        else:
            st.warning("This job is not a good fit for your CV :(")


if __name__ == "__main__":
    st.title("Document Reader")
    main()
