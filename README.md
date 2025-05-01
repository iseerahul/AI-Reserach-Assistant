# Advanced Research Assistant Chatbot

## Overview
it is a sophisticated research assistant chatbot that combines RAG (Retrieval-Augmented Generation) with document processing and research paper retrieval capabilities. Built with Google's Gemini model, it offers seamless interaction for academic research and document analysis.

## Features
- 📚 Research Paper Retrieval from Google Scholar and ArXiv
- 📄 PDF Document Processing and Summarization
- 🧠 RAG-based Knowledge Retrieval
- 💾 Conversation Memory System
- 🔍 Intelligent Context Understanding

## Setup

### Prerequisites
```bash
pip install -r equirements.txt
```

Required packages:
- google.generativeai
- python-dotenv
- faiss-cpu
- numpy
- sentence-transformers
- PyMuPDF
- aiohttp
- beautifulsoup4
- feedparser
- nltk

### Configuration
1. Create a `.env` file in the project root:
```plaintext
GEMINI_API_KEY=your_gemini_api_key_here
CORE_API_KEY=your_core_api_key_here
```

2. Create required folders:
```bash
mkdir research_papers
```

## Usage

### Starting the Chatbot
```bash
python chatbot2.0.py
```

### Key Commands
1. **Research Paper Search**
   ```
   find papers about machine learning
   search for AI research
   ```

2. **PDF Summarization**
   ```
   summarize filename.pdf
   tell me about document.pdf
   ```

3. **Document Q&A**
   ```
   what are the key points in the paper?
   explain the methodology in the research
   ```

4. **Basic Interaction**
   ```
   hello
   exit
   ```

## Project Structure
```
platy/
├── LLM_chatbot/
│   ├── chatbot2.0.py
│   └── faiss_index.faiss
├── research_papers/
│   └── [your PDF documents]
├── .env
└── requirements.txt
```

## Features in Detail

### 1. RAG System
- Uses FAISS for efficient similarity search
- Integrates with sentence transformers for embeddings
- Maintains conversation memory for context

### 2. Document Processing
- PDF text extraction and preprocessing
- Chunking system for large documents
- Cached processing for better performance

### 3. Research Paper Retrieval
- Concurrent fetching from multiple sources
- Deduplication of results
- PDF link prioritization

### 4. Memory System
- Long-term conversation storage
- Context-aware responses
- Relevance-based retrieval

## Performance Considerations
- Maximum chunk size: 600 words
- Embedding dimension: 768 (all-mpnet-base-v2)
- Response temperature: 0.5 for consistent outputs
- Configurable number of search results (default: 7)

## Error Handling
- Robust PDF processing with detailed error messages
- Network request timeout handling
- Graceful fallbacks for missing documents

## Contributing
Feel free to submit issues and enhancement requests.

## Acknowledgments
- Google Gemini API
- FAISS by Facebook Research
- SentenceTransformers
- PyMuPDF

