import os
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from docx import Document
import PyPDF2

# NLTK setup: auto-download punkt if missing
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Explicit English tokenizer to bypass punkt_tab issue
punkt_params = PunktParameters()
tokenizer = PunktSentenceTokenizer(punkt_params)

# File readers
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Text processing
def clean_text(text):
    """Remove extra spaces, tabs, newlines"""
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text

def chunk_text(text, max_words=200):
    """Split text into chunks of approximately max_words"""
    # Use explicit tokenizer to avoid punkt_tab errors
    sentences = tokenizer.tokenize(text)
    chunks = []
    current_chunk = ""
    word_count = 0

    for sent in sentences:
        words_in_sent = len(sent.split())
        if word_count + words_in_sent > max_words:
            chunks.append(current_chunk.strip())
            current_chunk = sent
            word_count = words_in_sent
        else:
            current_chunk += " " + sent
            word_count += words_in_sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
