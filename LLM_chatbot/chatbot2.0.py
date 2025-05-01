import os
import google.generativeai as genai
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for reading PDFs
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import feedparser
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict
import urllib.parse
from functools import lru_cache


# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab/english/")  # Added check for punkt_tab
except LookupError:
    nltk.download("punkt")
    nltk.download('punkt_tab')  # Download punkt_tab as well

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ Error: GEMINI_API_KEY not found in .env file!")

genai.configure(api_key=api_key)
core_api_key = os.getenv("CORE_API_KEY")

# Configure generation settings for more professional responses
generation_config = {
    "temperature": 0.5,  # Lower temperature for more deterministic responses
    "top_p": 0.9,
    "top_k": 30,  # Reduce top_k for focused responses
    "max_output_tokens": 500,  # Increase tokens for detailed explanations
    "response_mime_type": "text/plain",
}

# Initialize Model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)
chat_session = model.start_chat(history=[])

# Initialize Embedding Model for RAG & Memory
embedding_model = SentenceTransformer("all-mpnet-base-v2") # Switching to all-mpnet-base-v2 for better quality embeddings
embedding_dimension = 768 # Dimension for all-mpnet-base-v2
index = faiss.IndexFlatL2(embedding_dimension)

# Memory Index for Long-Term Chat Storage
memory_index = faiss.IndexFlatL2(embedding_dimension)
memory_store = []  # Store conversation history

doc_store = []  # Store (filename, chunk) tuples for retrieval
folder_path = "research_papers"

# Chunking parameters
MAX_CHUNK_SIZE = 600  # words, adjusting chunk size

# Constants for retrieval
GOOGLE_SCHOLAR_RESULTS = 7 # Increasing number of results fetched
ARXIV_RESULTS = 7

# ANSI color codes
BLUE = '\033[94m'
END = '\033[0m'

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def process_pdf(pdf_path: str) -> str:
    """Extract and preprocess text from PDF with enhanced error handling."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            raise ValueError("PDF contains no pages")
            
        text_content = []
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")
                text = text.replace('\n\n', ' ').strip()
                if text:
                    text_content.append(text)
            except Exception as e:
                print(f"Warning: Error processing page {page_num}: {e}")
                
        if not text_content:
            raise ValueError("No text could be extracted from PDF")
            
        return " ".join(text_content)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

@lru_cache(maxsize=32)
def process_pdf_cached(pdf_path: str) -> str:
    """Cached version of PDF processing for better performance."""
    return process_pdf(pdf_path)

# Chunk text into sentences, respecting max_chunk_size
def chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()  # Remove leading/trailing whitespace

        # Check if adding the sentence exceeds the chunk size limit
        if len((current_chunk + " " + sentence).split()) <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence  # Add space if not first sentence
        else:
            if current_chunk:  # Don't add empty chunk
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Index research papers
def index_documents(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
        return

    indexed_count = 0
    print(f"Scanning {folder_path} for documents...")
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".pdf", ".txt", ".docx")):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Indexing: {file_path}")
            
            try:
                if filename.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                if text:
                    chunks = chunk_text(text)
                    for chunk in chunks:
                        doc_store.append((filename, chunk))
                        embedding = embedding_model.encode([chunk])[0]
                        index.add(np.array([embedding]).astype('float32'))
                    indexed_count += 1
                    print(f"✅ Successfully indexed {filename}")
                else:
                    print(f"⚠️ No text extracted from {filename}")
            
            except Exception as e:
                print(f"❌ Error indexing {filename}: {str(e)}")

    print(f"Indexed {indexed_count} documents")

# Load existing FAISS index, or create a new one
def load_or_create_index(index_path: str, dimension: int) -> faiss.Index:
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print(f"✅ Loaded FAISS index from {index_path}")
            return index
        except Exception as e:
            print(f"Error loading index from {index_path}: {e}")
            print("Creating a new index...")
            return faiss.IndexFlatL2(dimension)
    else:
        print("Creating a new FAISS index...")
        return faiss.IndexFlatL2(dimension)

# Save the FAISS index to disk
def save_index(index: faiss.Index, index_path: str):
    try:
        faiss.write_index(index, index_path)
        print(f"💾 Saved FAISS index to {index_path}")
    except Exception as e:
        print(f"Error saving index to {index_path}: {e}")

# Index documents on startup
index_path = "faiss_index.faiss"  # Path to save/load the index
index = load_or_create_index(index_path, embedding_dimension)
index_documents(folder_path)

# Index chat memory
def index_memory_interaction(user_input, bot_response):
    interaction_text = f"User: {user_input}\nPlaty: {bot_response}"
    interaction_embedding = embedding_model.encode([interaction_text])[0]
    memory_index.add(np.array([interaction_embedding]))
    memory_store.append(interaction_text)

# Retrieve past chat context
def retrieve_memory_context(query, top_k=3):
    if memory_index.ntotal == 0:
        return ""
    query_embedding = embedding_model.encode([query])
    distances, indices = memory_index.search(np.array(query_embedding), top_k)
    retrieved_memories = [memory_store[i] for i in indices[0] if i < len(memory_store)]
    return "\n".join(retrieved_memories)

# Retrieve document-based context
def retrieve_relevant_context(query, top_k=4):
    if len(doc_store) == 0:
        return "No documents found in the knowledge base."
        
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), min(top_k, len(doc_store)))
    
    # Filter results by relevance threshold
    threshold = 0.7
    relevant_docs = []
    
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(doc_store):
            similarity_score = 1 / (1 + dist)  # Convert distance to similarity
            if similarity_score > threshold:
                filename, chunk = doc_store[idx]
                relevant_docs.append({
                    'filename': filename,
                    'chunk': chunk,
                    'score': similarity_score
                })
    
    if not relevant_docs:
        return "No relevant content found in the knowledge base."
        
    # Sort by relevance score and format response
    relevant_docs.sort(key=lambda x: x['score'], reverse=True)
    context = "\n\n".join([
        f"From {doc['filename']} (relevance: {doc['score']:.2f}):\n{doc['chunk']}"
        for doc in relevant_docs
    ])
    
    return context

# Asynchronous fetching
async def fetch_url(session, url):
    try:
        async with session.get(url, timeout=10, allow_redirects=True) as response:  # Add allow_redirects
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to extract direct PDF link from Google Scholar result
def extract_pdf_link_from_scholar_result(result):
    try:
        pdf_link = result.find('a', href=lambda href: href and href.endswith('.pdf'))['href']
        return pdf_link
    except:
        return None

# Fetch research papers from Google Scholar
async def fetch_scholar_papers(query, num_results=GOOGLE_SCHOLAR_RESULTS):
    search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&hl=en"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await fetch_url(session, search_url)

        if html is None:
            return []

        soup = BeautifulSoup(html, "html.parser")
        results = []
        for entry in soup.select(".gs_ri"):
            title_element = entry.select_one(".gs_rt a")
            if not title_element:
                continue
            title = title_element.text
            paper_link = title_element["href"]
            pdf_link = extract_pdf_link_from_scholar_result(entry)

            result = {
                "title": title,
                "link": paper_link,
                "pdf_link": pdf_link
            }
            results.append(result)

        return results[:num_results]
    except Exception as e:
        print(f"Error fetching from Google Scholar: {e}")
        return []

# Fetch research papers from ArXiv
async def fetch_arxiv_papers(query, num_results=ARXIV_RESULTS):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results={num_results}"
    try:
        async with aiohttp.ClientSession() as session:
            xml_content = await fetch_url(session, url)

        if xml_content is None:
            return []

        feed = feedparser.parse(xml_content)
        results = []
        for entry in feed.entries:
            result = {
                "title": entry.title,
                "link": entry.link,
                "pdf_link": next((link.href for link in entry.links if link.type == 'application/pdf'), None)
            }
            results.append(result)

        return results
    except Exception as e:
        print(f"Error fetching from ArXiv: {e}")
        return []

# Deduplicate and prioritize PDF links
def process_results(scholar_results, arxiv_results):
    all_results = scholar_results + arxiv_results
    deduplicated_results = []
    seen_titles = set()

    for result in all_results:
        title = result['title']
        if title not in seen_titles:
            deduplicated_results.append(result)
            seen_titles.add(title)

    # Sort by PDF link availability
    deduplicated_results.sort(key=lambda x: x['pdf_link'] is None) # Prioritize results with PDF links

    return deduplicated_results

def is_research_query(query: str) -> bool:
    """Determine if the query is research-related."""
    research_keywords = {
        'research', 'paper', 'study', 'article', 'publication',
        'find', 'search', 'author', 'journal', 'conference'
    }
    query_words = set(query.lower().split())
    return bool(query_words & research_keywords)

def format_response(query: str, context: str, papers: str = None) -> str:
    """Format the chatbot response based on query type."""
    if not context and not papers:
        return "I don't have enough information to answer your query. Please try rephrasing or ask another question."
        
    response = []
    
    if context:
        response.append("Based on the documents in my knowledge base:")
        response.append(context)
        
    if papers and is_research_query(query):
        response.append("\nRelevant research papers:")
        response.append(papers)
        
    return "\n\n".join(response)

def get_closest_pdf_match(query: str, pdfs: List[str]) -> Tuple[str, float]:
    """Find the closest matching PDF using fuzzy matching."""
    from difflib import SequenceMatcher
    
    best_match = None
    highest_ratio = 0
    
    query_words = set(query.lower().split())
    
    for pdf in pdfs:
        pdf_name = pdf.lower().replace('.pdf', '')
        # Check both exact and partial matches
        if pdf_name in query:
            return pdf, 1.0
            
        # Check word by word
        for word in query_words:
            ratio = SequenceMatcher(None, word, pdf_name).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = pdf
                
    return best_match, highest_ratio

async def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    
    # Enhanced PDF summarization trigger words and patterns
    summarize_patterns = [
        "summarize", "summarise", "summary", "explain", "tell me about",
        "what's in", "what is in", "what does", "show me", "analyze",
        "contents of", "details of", "information in"
    ]
    
    # Check for PDF-related query
    if any(pattern in user_input for pattern in summarize_patterns):
        pdfs = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        if not pdfs:
            return "No PDF documents found in the research papers folder."
        
        # Try to find the best matching PDF
        target_pdf, match_ratio = get_closest_pdf_match(user_input, pdfs)
        
        if target_pdf and match_ratio > 0.6:  # Threshold for acceptable match
            pdf_path = os.path.join(folder_path, target_pdf)
            print(f"📄 Summarizing: {target_pdf} (match confidence: {match_ratio:.2f})")
            
            pdf_text = process_pdf(pdf_path)
            if pdf_text:
                summary = await summarize_pdf_content(pdf_text)
                return f"# Summary of {target_pdf}\n\n{summary}"
            else:
                return f"⚠️ Could not extract text from {target_pdf}."
        
        # If no good match found but PDFs exist
        if len(pdfs) > 1:
            pdf_list = "\n".join([f"- {pdf}" for pdf in pdfs])
            return f"""# Available PDFs
Please specify which document you'd like me to summarize:

{pdf_list}

You can say "summarize [filename]" or "tell me about [filename]"."""
        else:
            # If only one PDF exists, use it
            target_pdf = pdfs[0]
            pdf_path = os.path.join(folder_path, target_pdf)
            print(f"📄 Using single available PDF: {target_pdf}")
            
            pdf_text = process_pdf(pdf_path)
            if pdf_text:
                summary = await summarize_pdf_content(pdf_text)
                return f"# Summary of {target_pdf}\n\n{summary}"
            else:
                return f"⚠️ Could not extract text from {target_pdf}."

    
    # Handle greetings without fetching papers
    greetings = {"hi", "hello", "hey", "hii", "hola"}
    if user_input in greetings:
        return """Hello! I'm your research assistant. I can help you with:
1. Finding relevant research papers
2. Answering questions about documents in my knowledge base
3. Summarizing PDF documents (just say 'summarize pdf' or 'summarize [filename]')"""

    if user_input in ["exit", "quit", "bye"]:
        return "Goodbye. Have a great day!"

    # Only fetch papers if the query seems research-related
    should_fetch_papers = any(keyword in user_input for keyword in [
        "research", "paper", "study", "article", "publication", "find", "search"
    ])

    memory_context = retrieve_memory_context(user_input)
    research_context = retrieve_relevant_context(user_input)

    paper_links = "No relevant papers found."
    if should_fetch_papers:
        # Fetch papers concurrently
        scholar_task = asyncio.create_task(fetch_scholar_papers(user_input))
        arxiv_task = asyncio.create_task(fetch_arxiv_papers(user_input))

        scholar_papers = await scholar_task
        arxiv_papers = await arxiv_task

        processed_papers = process_results(scholar_papers, arxiv_papers)
        paper_links = "\n".join([
            f"🔗 {paper['title']}: {BLUE}{paper['pdf_link'] if paper['pdf_link'] else paper['link']}{END}"
            for paper in processed_papers
        ])

    prompt = f"""
    You are Platy, a highly intelligent and helpful research assistant AI.
    You are designed to provide accurate and professional responses.
    You are an expert in various academic fields.

    Instructions:
    1. Provide detailed and informative answers to user questions.
    2. When possible, provide research paper links to support your answers.
    3. Maintain a professional tone, avoiding excessive emojis or casual language.
    4. If you don't know the answer or the information is not available, state that clearly and politely.
    5. Format your responses using Markdown for readability.
    6. Do NOT act like a chatbot or friendly assistant just give normal responses
    7. Use proper spacing between paragraphs and sentences for better readability
    8. Do not include any prefix like "Platy:" in your responses
    9. Format your response with proper line breaks and spacing
    10. When providing links in your response, format them in blue color using the format: {BLUE}link text{END}

    Context:
    - Chat History: {memory_context if memory_context else "None"}
    - Relevant Research Papers: {research_context if research_context else "None"}
    - Online Research Papers: {paper_links}

    User Input: {user_input}

    Response:
    """

    response = chat_session.send_message(prompt)
    formatted_response = response.text.strip()

    if "I don't know" in formatted_response or "I'm not sure" in formatted_response:
        formatted_response = "I am unable to provide an answer to this question based on available data."

    index_memory_interaction(user_input, formatted_response)
    return formatted_response

async def summarize_pdf_content(text: str) -> str:
    """Generate a summary of PDF content using Gemini."""
    if not text:
        return "Could not extract text from PDF."
        
    prompt = f"""
    Summarize the following text in a clear and concise manner. 
    Focus on the main points and key findings.
    
    Text to summarize:
    {text[:8000]}  # Limiting text length to avoid token limits
    
    Please provide:
    1. Main topic/purpose
    2. Key points
    3. Important findings/conclusions
    """
    
    try:
        response = chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary."

# Chat loop with asyncio
async def main():
    print("🤖 Platy (RAG-Enhanced with Memory & Research Fetching) is ready!")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye. Have a great day!")
            break
        response = await chatbot_response(user_input)
        print("\n" + response + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        save_index(index, index_path)  # Save the index on exit