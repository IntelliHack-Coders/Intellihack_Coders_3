import json
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from tqdm import tqdm
import time

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class SyntheticQAGenerator:
    def __init__(self, use_gpu=True):
        """Initialize the QA generator with required models"""
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Using device: {self.device}")
        
        # Use smaller, more efficient models
        qg_model_name = "valhalla/t5-small-qg-hl"
        summarizer_model_name = "sshleifer/distilbart-cnn-6-6"
        sentence_model_name = "all-MiniLM-L6-v2"
        
        try:
            # Question generation model
            print("Loading question generation model...")
            self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
            self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name)
            if self.device == "cuda":
                # Load model with half precision to save memory
                self.qg_model = self.qg_model.half().to(self.device)
            else:
                self.qg_model = self.qg_model.to(self.device)
                
            print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer(sentence_model_name)
            
            print("Loading summarization model...")
            self.summarizer = pipeline("summarization", model=summarizer_model_name, device=-1)  # Always on CPU
            
            # Initialize stopwords
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def extract_key_terms(self, text):
        """Extract key technical terms and concepts from text"""
        # Extract capitalized multi-word terms (likely technical terms)
        cap_terms = re.findall(r'\b(?:[A-Z][a-z]* )+[A-Z][a-z]*\b|\b[A-Z][a-zA-Z]+\b', text)
        
        sentences = sent_tokenize(text)
        noun_phrases = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            
            i = 0
            while i < len(pos_tags) - 1:
                if pos_tags[i][1].startswith('JJ') and pos_tags[i+1][1].startswith('NN'):
                    noun_phrases.append(f"{pos_tags[i][0]} {pos_tags[i+1][0]}")
                i += 1
        
        numeric_terms = re.findall(r'\b[A-Za-z]+\d+\b', text)
        
        # Combine all terms and remove duplicates
        all_terms = list(set(cap_terms + noun_phrases + numeric_terms))
        
        # Filter out very short terms and stopwords
        filtered_terms = [term for term in all_terms if len(term) > 2 and not all(word.lower() in self.stop_words for word in term.split())]
        
        return filtered_terms

    def extract_key_sentences(self, text, num_sentences=5):
        """Extract the most informative sentences from the text"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        embeddings = self.sentence_model.encode(sentences)
        
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate similarity to centroid
        similarities = []
        for emb in embeddings:
            similarity = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            similarities.append(similarity)
        
        # Get top sentences
        top_indices = np.argsort(similarities)[-num_sentences:]
        top_indices = sorted(top_indices) 
        
        return [sentences[i] for i in top_indices]

    def generate_question_from_text(self, context, max_length=64):
        """Generate a question based on the context with error handling"""
        try:
            if len(context.split()) > 100:
                context = " ".join(context.split()[:100])
                
            prompt = f"generate question: {context}"
            
            inputs = self.qg_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.qg_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_beams=2,  # Reduce beam search to save memory
                    early_stopping=True
                )
                
            question = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the question
            question = question.strip()
            if not question.endswith('?'):
                question += '?'
                
            # Move tensors to CPU and clear memory
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return question
            
        except Exception as e:
            print(f"Error generating question: {e}")
            # Fallback to a simple question
            words = context.split()
            if len(words) > 5:
                subject = " ".join(words[:3]) + "..."
                return f"What can you tell me about {subject}?"
            return "What is the main point of this text?"

    def generate_template_questions(self, title, terms):
        """Generate questions using templates"""
        questions = []
        
        # Add title-based questions
        title_templates = [
            f"What is {title} about?",
            f"How does {title} work?",
            f"What problem does {title} solve?",
            f"What are the key features of {title}?",
            f"Explain the concept of {title}."
        ]
        questions.extend(random.sample(title_templates, min(2, len(title_templates))))
        
        # Generate term-based questions
        for term in terms[:min(3, len(terms))]:
            term_templates = [
                f"What is {term} in the context of {title}?",
                f"How is {term} used in {title}?",
                f"Why is {term} important for {title}?",
                f"Explain the concept of {term} as mentioned in {title}."
            ]
            questions.append(random.choice(term_templates))
        
        return questions

    def generate_concise_answer(self, text, max_length=150):
        """Generate a concise answer from the text with error handling"""
        if len(text.split()) < 50:  # If text is already short
            return text
            
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error summarizing text: {e}")
            # Fallback: return the first few sentences
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:min(3, len(sentences))])

    def generate_qa_pairs(self, chunks, max_pairs_per_chunk=8):
        """Generate QA pairs from document chunks with improved memory management"""
        qa_pairs = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            try:
                title = chunk["title"]
                text = chunk["text"]
                
                terms = self.extract_key_terms(text)
                key_sentences = self.extract_key_sentences(text)
                
                template_questions = self.generate_template_questions(title, terms)
                
                model_questions = []
                for sentence in key_sentences[:2]:  
                    question = self.generate_question_from_text(sentence)
                    model_questions.append(question)
                    if self.device == "cuda":
                        time.sleep(0.1)
                        
                # Combine questions
                all_questions = template_questions + model_questions
                random.shuffle(all_questions)
                selected_questions = all_questions[:max_pairs_per_chunk]
                
                # Generate answers
                for question in selected_questions:
                    # Create a concise answer
                    answer = self.generate_concise_answer(text)
                    
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "chunk_id": i,
                        "title": title
                    })
                    
                if i % 5 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        return qa_pairs

if __name__ == "__main__":
    generator = SyntheticQAGenerator(use_gpu=True) 
    
    with open("./document_chunks.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    
    qa_pairs = generator.generate_qa_pairs(all_chunks, max_pairs_per_chunk=5)  # Reduced from 8 to 5
    
    # Split into train/validation/test sets
    train_data, temp_data = train_test_split(qa_pairs, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Created {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test examples")
    
    # Save datasets
    with open('./train_data2.json', 'w') as f:
        json.dump(train_data, f)
    with open('./val_data2.json', 'w') as f:
        json.dump(val_data, f)
    with open('./test_data2.json', 'w') as f:
        json.dump(test_data, f)