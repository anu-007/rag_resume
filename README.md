# RAG Resume Tailoring Tool

This project is a RAG (Retrieval-Augmented Generation) based tool that helps you tailor your resume to a specific job description. It takes your resume and a job description as input, and generates an ATS-friendly resume, a cover letter, a professional email, and a direct message for recruiters.

## Features

-   **Resume Tailoring:** Automatically tailors your resume to match the requirements of a specific job description.
-   **ATS-Friendly:** Generates a resume that is optimized for Applicant Tracking Systems (ATS).
-   **Multiple Outputs:** Creates a cover letter, a professional email, and a direct message in addition to the tailored resume.
-   **RAG-based:** Uses a Retrieval-Augmented Generation pipeline to ensure that the generated content is relevant and accurate.

## How it Works

The project uses a sophisticated RAG pipeline to generate the tailored documents. Here's a breakdown of the pipeline:

1.  **Loading:** The tool loads your resume (in PDF or text format) and the job description (from a URL).
2.  **Chunking:** The loaded documents are split into smaller chunks to facilitate efficient processing.
3.  **Retrieval:** The tool uses a hybrid retrieval system that combines three different retrieval methods:
    *   **BM25:** A keyword-based retrieval method that is effective at finding documents with matching keywords.
    *   **Chroma:** A semantic retrieval method that uses embeddings to find documents that are semantically similar to the query.
    *   **HyDE:** A method that generates a hypothetical document and then uses it to retrieve relevant documents.
4.  **Reranking:** The retrieved documents are reranked using a Flashrank reranker to ensure that the most relevant documents are at the top.
5.  **Generation:** The reranked documents are then used as context for a large language model (LLM) to generate the tailored resume, cover letter, email, and direct message.

## Usage

To use the tool, you need to provide your resume, the job description URL, and a question. The tool will then generate the tailored documents and save them in the `out` directory.

```bash
python main.py --resume <path_to_your_resume> --jd <job_description_url> --question "Create ATS resume, cover, email, and DM for this job."
```

## Dependencies

The project uses the following libraries:

-   `langchain`
-   `langchain-community`
-   `langchain-ollama`
-   `beautifulsoup4`
-   `requests`
-   `chromadb`
-   `flashrank`
-   `argparse`

You can install these dependencies using pip:

```bash
pip install langchain langchain-community langchain-ollama beautifulsoup4 requests chromadb flashrank argparse
```
