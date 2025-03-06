# LLM Question-Answering Application

This project is a Streamlit-based application that allows users to upload documents (PDF, DOCX, TXT), split them into chunks, create embeddings using OpenAI, and perform question-answering using a language model.

## Features

- Upload and process PDF, DOCX, and TXT files
- Split documents into chunks
- Create embeddings using OpenAI's `text-embedding-3-small` model
- Perform question-answering using a language model
- Display chat history

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up the OpenAI API key:

    - Create a `.env` file in the project root directory and add your OpenAI API key:

        ```env
        OPENAI_API_KEY=your_openai_api_key
        ```

## Usage

1. Run the Streamlit application:

    ```sh
    streamlit run chat_with_documents_01.py
    ```

2. Open the application in your web browser.

3. Upload a document (PDF, DOCX, or TXT) using the file uploader in the sidebar.

4. Adjust the chunk size and `k` value as needed.

5. Click the "Add Data" button to process the uploaded document.

6. Ask a question about the content of the uploaded document using the text input field.

## File Structure

- `chat_with_documents_01.py`: Main application script
- `requirements.txt`: List of required Python packages
- `img.png`: Image displayed in the application

## Dependencies

- Python 3.10
- Streamlit
- LangChain
- OpenAI
- Chroma
- tiktoken
- python-dotenv

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
