# LLM Question-Answering Application

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)](https://streamlit.io/)

A professional, open-source Streamlit application for interactive question-answering over your own documents using OpenAI embeddings and LLMs.

---

## Features

- Upload and process PDF, DOCX, and TXT files
- Split documents into semantic chunks
- Create embeddings using OpenAI's `text-embedding-3-small` model
- Perform question-answering using a language model (GPT-3.5)
- Display chat history for context
- Simple, modern UI

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/llm-question-answering-app.git
    cd llm-question-answering-app
    ```
2. **Create a virtual environment and activate it:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
4. **Set up the OpenAI API key:**
    - Copy `.env.example` to `.env` and add your OpenAI API key:
      ```env
      OPENAI_API_KEY=your_openai_api_key
      ```

---

## Usage

1. **Run the Streamlit application:**
    ```sh
    streamlit run chat_with_documents_01.py
    ```
2. **Open the app in your browser** (Streamlit will provide a local URL).
3. **Upload a document** (PDF, DOCX, or TXT) using the sidebar.
4. **Adjust chunk size and `k` value** as needed.
5. **Click "Add Data"** to process the document.
6. **Ask questions** about your document using the input field.

---

## Screenshot

![App Screenshot](img.png)

---

## File Structure

- `chat_with_documents_01.py`: Main application script
- `requirements.txt`: Python dependencies
- `img.png`: App logo/screenshot
- `.env.example`: Example environment file

---

## Dependencies

- Python 3.10+
- Streamlit
- LangChain
- OpenAI
- Chroma
- tiktoken
- python-dotenv

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## Attribution

This project is inspired by and based on the course "Developing LLM Apps with LangChain" from [Zero To Mastery Academy](https://zerotomastery.io/).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
