# RAG Document Q&A System

This is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. The system uses LangChain, InMemoryVectorStore, and OpenAI's GPT models to provide accurate answers based on the document content.

## Features

- PDF document upload and processing
- Document chunking and embedding
- Question answering using RAG
- User-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

## Usage

1. Upload PDF documents using the file uploader in the sidebar
2. Wait for the documents to be processed
3. Enter your question in the text input field
4. View the generated answer

## Deployment

### Local Deployment
The application can be run locally using Streamlit's built-in server.

### Cloud Deployment
The application can be deployed to various cloud platforms:

#### Streamlit Cloud
1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the application

#### Heroku
1. Create a `Procfile`:
```
web: streamlit run app.py
```

2. Deploy to Heroku:
```bash
heroku create
git push heroku main
```

#### Docker Deployment
1. Build the Docker image:
```bash
docker build -t rag-app .
```

2. Run the container:
```bash
docker run -p 8501:8501 rag-app
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key

## License

MIT License 
