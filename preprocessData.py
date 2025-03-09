import os
import re
import json
import chardet
import fitz  # PyMuPDF
from tqdm import tqdm

def read_documents(directory):
    """Read all documents from a directory"""
    documents = []
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if filename.endswith('.pdf'):
            # Handle PDF files
            try:
                text = ""
                with fitz.open(filepath) as pdf_doc:
                    for page in pdf_doc:
                        text += page.get_text()
                documents.append({
                    "title": filename.split('.')[0],
                    "content": text,
                    "source": filename
                })
            except Exception as e:
                print(f"Failed to process PDF {filename}: {e}")
        
        elif filename.endswith('.md'):
            # Handle Markdown files
            try:
                with open(filepath, 'rb') as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding = 'utf-8'  # Default to UTF-8 if encoding is None
                
                with open(filepath, 'r', encoding=encoding) as file:
                    content = file.read()
                    documents.append({
                        "title": filename.split('.')[0],
                        "content": content,
                        "source": filename
                    })
            except UnicodeDecodeError:
                print(f"Failed to decode {filename} with detected encoding {encoding}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    return documents

def chunk_document(document, chunk_size=500, overlap=100):
    """Split document into overlapping chunks"""
    content = document["content"]
    
    # Try to split at paragraph or section boundaries first
    paragraphs = re.split(r'\n\s*\n', content)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk and start a new one
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "source": document["source"],
                "title": document["title"]
            })
            # Keep some overlap from the previous chunk
            last_sentences = " ".join(current_chunk.split(". ")[-3:])
            current_chunk = last_sentences + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "source": document["source"],
            "title": document["title"]
        })
    
    return chunks

# Process all documents
documents = read_documents(r"C:\Users\REDTECH\Desktop\competition")
all_chunks = []

for doc in documents:
    chunks = chunk_document(doc)
    all_chunks.extend(chunks)

print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

# Save chunks for later use
with open('document_chunks_new.json', 'w') as f:
    json.dump(all_chunks, f)