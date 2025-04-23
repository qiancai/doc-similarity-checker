#!/usr/bin/env python3
"""
Document Similarity Checker
This script compares two sets of markdown documents to find similar content.
It uses sentence-transformers to generate embeddings and compute similarities.
"""

import os
import argparse
import glob
import re
import time
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import string
from collections import Counter
import pandas as pd

# Set TOKENIZERS_PARALLELISM environment variable to prevent warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, util
import torch
import pickle

# Configuration variables - modify these as needed
DOC_SET_1 = "/Users/grcai/Documents/GitHub/doc-similarity-checker/for-test/doc-set-1"  # Path to the first document set
DOC_SET_2 = "/Users/grcai/Documents/GitHub/doc-similarity-checker/for-test/doc-set-2"  # Path to the second document set
MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model name for first-stage filtering
SECOND_MODEL_NAME = "all-mpnet-base-v2"  # More accurate model for second-stage verification
SIMILARITY_THRESHOLD = 0.60  # Similarity threshold (first-stage filtering)
MPNET_SIMILARITY_THRESHOLD = 0.80  # Second-stage verification threshold
BERT_WEIGHT = 0.7  # BERT similarity weight for hybrid similarity
JACCARD_WEIGHT = 0.3  # Jaccard similarity weight for hybrid similarity
SEGMENT_TYPE = "sentence"  # Text segmentation method ('paragraph' or 'sentence')
BATCH_SIZE = 64  # Batch size for encoding
OUTPUT_FILE = "check_results.md"  # Output file for results
SHORT_SENTENCES_OUTPUT_FILE = "check_results_short_sentences.md"  # Output file for short sentences results
EXCEL_OUTPUT_FILE = "similarity_results.xlsx"  # Excel output file for results
IGNORE_FILE = "ignore.txt"  # File containing text patterns to ignore
USE_GPU = torch.cuda.is_available()  # Use GPU if available
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Number of worker processes for parallel processing
EMBEDDING_CACHE_1 = "cache/doc_set_1_embeddings.pkl"  # Path to save/load embeddings for document set 1
EMBEDDING_CACHE_2 = "cache/doc_set_2_embeddings.pkl"  # Path to save/load embeddings for document set 2
USE_CACHE = False  # Whether to use embedding cache (True: use if available, False: always recalculate)
MAX_BATCH_FILES = 50  # Maximum number of files to process in each batch
CHUNKED_PROCESSING = True  # Enable chunked processing for large document sets
MAX_CACHE_SIZE_GB = 2  # Maximum cache file size (GB)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check similarity between two document sets")
    parser.add_argument("--doc-set-1", type=str, default=DOC_SET_1, help="Path to the first document set")
    parser.add_argument("--doc-set-2", type=str, default=DOC_SET_2, help="Path to the second document set")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="SentenceTransformer model name for first-stage filtering")
    parser.add_argument("--second-model", type=str, default=SECOND_MODEL_NAME, 
                       help="SentenceTransformer model name for second-stage verification")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, 
                       help="Similarity threshold for first-stage filtering (0-1)")
    parser.add_argument("--mpnet-threshold", type=float, default=MPNET_SIMILARITY_THRESHOLD, 
                       help="Similarity threshold for second-stage verification (0-1)")
    parser.add_argument("--bert-weight", type=float, default=BERT_WEIGHT, 
                      help="Weight for BERT similarity in hybrid approach (0-1)")
    parser.add_argument("--jaccard-weight", type=float, default=JACCARD_WEIGHT, 
                      help="Weight for Jaccard similarity in hybrid approach (0-1)")
    parser.add_argument("--segment", choices=["paragraph", "sentence"], default=SEGMENT_TYPE, 
                       help="Text segmentation method")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for encoding")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file for results")
    parser.add_argument("--short-output", type=str, default=SHORT_SENTENCES_OUTPUT_FILE, 
                       help="Output file for short sentences results")
    parser.add_argument("--excel-output", type=str, default=EXCEL_OUTPUT_FILE, 
                       help="Excel output file for all results")
    parser.add_argument("--ignore-file", type=str, default=IGNORE_FILE,
                       help="File containing text patterns to ignore")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process per document set")
    parser.add_argument("--use-gpu", action="store_true", default=USE_GPU, help="Use GPU for embedding generation")
    parser.add_argument("--embedding-cache-1", type=str, default=EMBEDDING_CACHE_1, 
                       help="Path to save/load embeddings for document set 1")
    parser.add_argument("--embedding-cache-2", type=str, default=EMBEDDING_CACHE_2, 
                       help="Path to save/load embeddings for document set 2")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Do not use embedding cache even if available (always recalculate)")
    parser.add_argument("--chunked", action="store_true", default=CHUNKED_PROCESSING, 
                       help="Enable chunked processing for large document sets")
    parser.add_argument("--max-batch-files", type=int, default=MAX_BATCH_FILES, 
                       help="Maximum number of files to process in each batch when chunked processing is enabled")
    parser.add_argument("--max-cache-size", type=float, default=MAX_CACHE_SIZE_GB, 
                       help="Maximum cache file size in GB")
    return parser.parse_args()

def find_markdown_files(directory: str, max_files: int = None) -> List[str]:
    """
    Find all markdown files in the given directory and its subdirectories
    
    Args:
        directory: Path to the directory to search
        max_files: Maximum number of files to return (for testing purposes)
        
    Returns:
        List of paths to markdown files
    """
    files = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
    if max_files is not None and len(files) > max_files:
        return files[:max_files]
    return files

def read_markdown_file(file_path: str) -> str:
    """
    Read the content of a markdown file and return as a string
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Content of the file as a string
    """
    # Try multiple encodings to handle various file formats
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read {file_path} using {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed to decode {file_path} with {encoding} encoding, trying next...")
            continue
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return ""
        except Exception as e:
            print(f"Error reading {file_path} with {encoding} encoding: {e}")
            continue
    
    if content is None:
        print(f"Failed to read {file_path} with any encoding, returning empty string")
        return ""
    
    return content

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs
    
    Args:
        text: The text to split
        
    Returns:
        List of paragraphs
    """
    # Split by double newlines and filter out empty paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text)]
    return [p for p in paragraphs if p]

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences with improved handling for Markdown content
    
    Args:
        text: The text to split
        
    Returns:
        List of sentences
    """
    # Temporarily replace certain patterns that could confuse sentence splitting
    # Protect URLs, file paths, and version numbers
    protected_patterns = []
    
    # Protect URLs
    url_pattern = re.compile(r'https?://[^\s]+')
    matches = url_pattern.finditer(text)
    for i, match in enumerate(matches):
        placeholder = f"___URL_{i}___"
        protected_patterns.append((placeholder, match.group(0)))
        text = text.replace(match.group(0), placeholder)
    
    # Protect version numbers and file paths
    version_pattern = re.compile(r'\b\d+\.\d+(\.\d+)*\b')
    matches = version_pattern.finditer(text)
    for i, match in enumerate(matches):
        placeholder = f"___VERSION_{i}___"
        protected_patterns.append((placeholder, match.group(0)))
        text = text.replace(match.group(0), placeholder)
    
    # Split sentences using common end punctuation
    # Handle cases like "e.g." and "i.e." as non-sentence boundaries
    text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', '\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    # Replace placeholders back
    restored_sentences = []
    for sentence in sentences:
        for placeholder, original in protected_patterns:
            sentence = sentence.replace(placeholder, original)
        restored_sentences.append(sentence)
    
    return restored_sentences

def extract_segments(file_path: str, segment_type: str) -> List[Tuple[str, str]]:
    """
    Extract text segments from a markdown file
    
    Args:
        file_path: Path to the markdown file
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        
    Returns:
        List of tuples containing (file_path, segment_text)
    """
    text = read_markdown_file(file_path)
    
    # First split text by lines to avoid comparing multi-line segments with mixed content
    lines = text.split('\n')
    
    # Process each line or group of lines as appropriate
    filtered_segments = []
    in_code_block = False
    current_paragraph = []
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            # Reset paragraph when entering or exiting a code block
            current_paragraph = []
            continue
        
        if in_code_block:
            continue
        
        # Skip empty lines and markdown separators
        line = line.strip()
        if not line or line.startswith('---') or re.match(r'^[\|\-\s]+$', line):
            # End of paragraph - process what we've collected if anything
            if current_paragraph and segment_type == "paragraph":
                joined_paragraph = " ".join(current_paragraph).strip()
                if len(joined_paragraph.split()) >= 5:
                    filtered_segments.append(joined_paragraph)
                current_paragraph = []
            continue
        
        # Clean the line for consistent processing
        clean_line = clean_markdown_line(line)
        
        # For paragraph segmentation, collect lines
        if segment_type == "paragraph":
            current_paragraph.append(clean_line)
        else:  # For sentence segmentation
            # Split the line into sentences
            line_sentences = split_into_sentences(clean_line)
            for sentence in line_sentences:
                if len(sentence.split()) >= 5:
                    filtered_segments.append(sentence)
    
    # Don't forget to process any remaining paragraph
    if current_paragraph and segment_type == "paragraph":
        joined_paragraph = " ".join(current_paragraph).strip()
        if len(joined_paragraph.split()) >= 5:
            filtered_segments.append(joined_paragraph)
    
    return [(file_path, segment) for segment in filtered_segments]

def clean_markdown_line(line: str) -> str:
    """
    Clean a single line of markdown text
    
    Args:
        line: The line to clean
        
    Returns:
        Cleaned line
    """
    clean_line = line.strip()
    
    # Remove markdown headings
    if clean_line.startswith('#'):
        clean_line = re.sub(r'^#+\s+', '', clean_line)
    
    # Clean list markers
    clean_line = re.sub(r'^[\*\-\+]\s+', '', clean_line)
    
    # Remove HTML tags
    clean_line = re.sub(r'<[^>]*>', '', clean_line)
    
    # Remove markdown links but keep the text
    clean_line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_line)
    
    # Remove inline code markers
    clean_line = re.sub(r'`([^`]+)`', r'\1', clean_line)
    
    # Remove common markdown annotations [text]{#annotation}
    clean_line = re.sub(r'\{#[a-zA-Z0-9_-]+\}', '', clean_line)
    
    return clean_line

def extract_segments_safe(file_path: str, segment_type: str) -> List[Tuple[str, str]]:
    """
    Safely extract text segments from a markdown file, catching and handling exceptions
    
    Args:
        file_path: Path to the markdown file
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        
    Returns:
        List of tuples containing (file_path, segment_text) or empty list if there was an error
    """
    try:
        return extract_segments(file_path, segment_type)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def process_document_set(directory: str, segment_type: str, max_files: int = None, 
                         num_workers: int = NUM_WORKERS) -> List[Tuple[str, str]]:
    """
    Process all markdown files in a document set
    
    Args:
        directory: Path to the document set
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        max_files: Maximum number of files to process (for testing)
        num_workers: Number of worker processes
        
    Returns:
        List of tuples containing (file_path, segment_text)
    """
    markdown_files = find_markdown_files(directory, max_files)
    all_segments = []
    
    print(f"Processing {len(markdown_files)} markdown files...")
    
    # For small number of files, process sequentially
    if len(markdown_files) < 10 or num_workers <= 1:
        for file_path in tqdm(markdown_files, desc="Processing files"):
            try:
                segments = extract_segments(file_path, segment_type)
                all_segments.extend(segments)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                print(f"Skipping file {file_path}")
    else:
        # For larger sets, use parallel processing
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.starmap(
                    extract_segments_safe, 
                    [(file_path, segment_type) for file_path in markdown_files]
                ),
                total=len(markdown_files),
                desc="Processing files in parallel"
            ))
            for result in results:
                all_segments.extend(result)
    
    print(f"Extracted {len(all_segments)} segments from {len(markdown_files)} files")
    return all_segments

def compute_embeddings(model: SentenceTransformer, segments: List[Tuple[str, str]], 
                       batch_size: int) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    """
    Compute embeddings for text segments
    
    Args:
        model: SentenceTransformer model
        segments: List of (file_path, segment_text) tuples
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (embeddings, segments)
    """
    texts = [segment[1] for segment in segments]
    
    # Show progress bar for embedding generation
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        convert_to_tensor=True,
        show_progress_bar=True
    )
    
    return embeddings, segments

def process_similarity_chunk(start_idx, end_idx, embeddings1, segments1, embeddings2, segments2, threshold):
    """
    Process a similarity calculation chunk
    
    Args:
        start_idx: Start index
        end_idx: End index
        embeddings1: Embeddings from the first document set
        segments1: Segments from the first document set
        embeddings2: Embeddings from the second document set
        segments2: Segments from the second document set
        threshold: Similarity threshold
        
    Returns:
        List of similar pairs
    """
    embeddings_chunk = embeddings1[start_idx:end_idx]
    segments_chunk = segments1[start_idx:end_idx]
    
    # Calculate cosine similarity for the current chunk
    cosine_scores = util.cos_sim(embeddings_chunk, embeddings2)
    
    # Find pairs with similarity above threshold
    similar_pairs = []
    for chunk_idx, scores in enumerate(cosine_scores):
        orig_idx = start_idx + chunk_idx
        for j, score in enumerate(scores):
            if score >= threshold:
                similar_pairs.append((segments_chunk[chunk_idx], segments2[j], score.item()))
    
    return similar_pairs

def has_contextual_relevance(text1: str, text2: str) -> bool:
    """
    Check if two text segments have contextual relevance (not just similar instruction patterns)
    
    Args:
        text1: First text segment
        text2: Second text segment
        
    Returns:
        True if the text segments have contextual relevance, False otherwise
    """
    # Convert to lowercase and tokenize
    words1 = set(text1.lower().translate(str.maketrans('', '', string.punctuation)).split())
    words2 = set(text2.lower().translate(str.maketrans('', '', string.punctuation)).split())
    
    # Get keywords excluding stop words
    stop_words = {'the', 'this', 'that', 'and', 'or', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'you', 'can', 'is', 'are', 'be'}
    keywords1 = words1 - stop_words
    keywords2 = words2 - stop_words
    
    # If both are UI instruction patterns but share fewer than 2 keywords, likely false similarity
    if (('click' in words1 or 'select' in words1) and ('click' in words2 or 'select' in words2)):
        common_keywords = keywords1.intersection(keywords2)
        if len(common_keywords) < 2:
            return False
    
    # Get technical terms and entity nouns (assuming words >= 5 chars are more likely to be technical terms)
    terms1 = {w for w in keywords1 if len(w) >= 5}
    terms2 = {w for w in keywords2 if len(w) >= 5}
    
    # If they share important terms, they're relevant
    if terms1.intersection(terms2):
        return True
    
    # Extract possible code snippets or configurations (words with special characters)
    code_pattern = re.compile(r'[a-zA-Z0-9_]+[._\-][a-zA-Z0-9_]+')
    code1 = set(re.findall(code_pattern, text1))
    code2 = set(re.findall(code_pattern, text2))
    
    # If they share code snippets, they're relevant
    if code1.intersection(code2):
        return True
    
    # Top nouns
    nouns1 = [w for w in keywords1 if len(w) > 3]
    nouns2 = [w for w in keywords2 if len(w) > 3]
    
    # If they share very few main nouns, might not be relevant
    common_nouns = set(nouns1).intersection(set(nouns2))
    if len(common_nouns) < 2 and (len(nouns1) >= 3 and len(nouns2) >= 3):
        return False
    
    return True

def is_ui_instruction(text: str) -> bool:
    """
    Detect if text is a UI instruction (like click, edit, etc.)
    
    Args:
        text: Text to check
        
    Returns:
        True if it's a UI instruction, False otherwise
    """
    # List of UI interaction verbs
    ui_verbs = ['click', 'tap', 'select', 'choose', 'edit', 'modify', 'press', 'check']
    
    # Look for common UI interaction patterns
    text_lower = text.lower()
    
    # If text is short and contains a UI interaction verb, it's more likely to be a UI instruction
    if len(text.split()) < 10:
        for verb in ui_verbs:
            if verb in text_lower:
                return True
    
    # Look for common UI instruction patterns
    ui_patterns = [
        r'click\s+(on\s+)?(the\s+)?["`\']?[\w\s]+["`\']?',
        r'(press|select|choose)\s+(the\s+)?["`\']?[\w\s]+["`\']?',
        r'go\s+to\s+(the\s+)?["`\']?[\w\s]+["`\']?',
    ]
    
    for pattern in ui_patterns:
        if re.search(pattern, text_lower):
            return True
            
    return False

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two text strings
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Jaccard similarity score (0-1)
    """
    # Normalize text: lowercase and remove punctuation
    def normalize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    # Get word sets
    set1 = set(normalize(text1).split())
    set2 = set(normalize(text2).split())
    
    # Calculate Jaccard similarity
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_hybrid_similarity(bert_similarity: float, text1: str, text2: str, 
                              bert_weight: float = BERT_WEIGHT, 
                              jaccard_weight: float = JACCARD_WEIGHT) -> float:
    """
    Calculate hybrid similarity by combining BERT and Jaccard similarities
    
    Args:
        bert_similarity: Pre-calculated BERT similarity
        text1: First text string
        text2: Second text string
        bert_weight: Weight for BERT similarity (default: 0.7)
        jaccard_weight: Weight for Jaccard similarity (default: 0.3)
        
    Returns:
        Hybrid similarity score (0-1)
    """
    # Calculate Jaccard similarity
    jaccard_similarity = calculate_jaccard_similarity(text1, text2)
    
    # Calculate hybrid similarity
    hybrid_sim = bert_weight * bert_similarity + jaccard_weight * jaccard_similarity
    
    return hybrid_sim

def find_similar_segments_parallel(segments1: List[Tuple[str, str]], embeddings1: torch.Tensor,
                        segments2: List[Tuple[str, str]], embeddings2: torch.Tensor, 
                        threshold: float, num_workers: int = NUM_WORKERS,
                        bert_weight: float = BERT_WEIGHT,
                        jaccard_weight: float = JACCARD_WEIGHT) -> List[Tuple[Tuple[str, str], Tuple[str, str], float]]:
    """
    Find similar segments between two document sets using parallel processing and hybrid similarity
    
    Args:
        segments1: List of (file_path, segment_text) tuples from the first document set
        embeddings1: Embeddings for segments1
        segments2: List of (file_path, segment_text) tuples from the second document set
        embeddings2: Embeddings for segments2
        threshold: Similarity threshold
        num_workers: Number of worker processes
        bert_weight: Weight for BERT similarity
        jaccard_weight: Weight for Jaccard similarity
        
    Returns:
        List of (segment1, segment2, similarity_score) tuples
    """
    print("Computing cosine similarities using parallel processing...")
    start_time = time.time()
    
    # Split large embedding tensors into chunks
    chunk_size = min(1000, max(100, len(embeddings1) // (num_workers * 2)))
    chunks = [(i, min(i + chunk_size, len(embeddings1))) for i in range(0, len(embeddings1), chunk_size)]
    
    print(f"Splitting {len(embeddings1)} embeddings into {len(chunks)} chunks for parallel processing (about {chunk_size} embeddings per chunk)")
    
    # For large document sets, use chunked non-parallel processing instead of multiprocessing
    # This avoids overhead from serialization and inter-process communication
    print("Starting chunked similarity calculation...")
    all_similar_pairs = []
    
    # Ensure all data is on CPU
    if isinstance(embeddings1, torch.Tensor) and embeddings1.is_cuda:
        print("Moving first embedding set from GPU to CPU...")
        embeddings1 = embeddings1.cpu()
        
    if isinstance(embeddings2, torch.Tensor) and embeddings2.is_cuda:
        print("Moving second embedding set from GPU to CPU...")
        embeddings2 = embeddings2.cpu()
    
    # Use tqdm to show progress
    for start_idx, end_idx in tqdm(chunks, desc="Processing similarity chunks"):
        # Get current chunk
        embeddings1_chunk = embeddings1[start_idx:end_idx]
        segments1_chunk = segments1[start_idx:end_idx]
        
        # Calculate cosine similarity for current chunk
        cosine_scores = util.cos_sim(embeddings1_chunk, embeddings2)
        
        # Find pairs with similarity above threshold
        for chunk_idx, scores in enumerate(cosine_scores):
            orig_idx = start_idx + chunk_idx
            
            # Get current segment
            current_segment_text = segments1_chunk[chunk_idx][1]
            
            # Use vectorized operation to find indices above threshold (using a lower initial threshold for filtering)
            initial_threshold = max(0.4, threshold - 0.15)  # Initial threshold lower than final to capture potential matches
            above_threshold = torch.where(scores >= initial_threshold)[0]
            
            for j in above_threshold:
                j_idx = j.item()  # Convert to Python integer
                matched_segment_text = segments2[j_idx][1]
                bert_similarity = scores[j_idx].item()
                
                # Calculate hybrid similarity
                hybrid_sim = calculate_hybrid_similarity(
                    bert_similarity, 
                    current_segment_text, 
                    matched_segment_text, 
                    bert_weight, 
                    jaccard_weight
                )
                
                # For UI instruction texts, use a higher similarity threshold
                if is_ui_instruction(current_segment_text) and is_ui_instruction(matched_segment_text):
                    # Require higher similarity for UI instructions
                    if hybrid_sim >= (threshold + 0.05):
                        all_similar_pairs.append((segments1_chunk[chunk_idx], segments2[j_idx], hybrid_sim))
                else:
                    # Use original threshold for non-UI instruction text
                    if hybrid_sim >= threshold:
                        all_similar_pairs.append((segments1_chunk[chunk_idx], segments2[j_idx], hybrid_sim))
        
        # Free memory
        del cosine_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Similarity calculation complete, sorting {len(all_similar_pairs)} results...")
    # Sort by similarity (descending)
    all_similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    elapsed_time = time.time() - start_time
    print(f"Found {len(all_similar_pairs)} similar pairs, processing time: {elapsed_time:.2f} seconds")
    
    return all_similar_pairs

def save_embeddings(file_path: str, embeddings: torch.Tensor, segments: List[Tuple[str, str]], 
                   doc_dir: str, model_name: str, segment_type: str, max_cache_size_gb: float = MAX_CACHE_SIZE_GB):
    """
    Save embeddings and segments to a pickle file
    
    Args:
        file_path: Path to save the embeddings
        embeddings: Tensor of embeddings
        segments: List of (file_path, segment_text) tuples
        doc_dir: Path to the document directory
        model_name: Name of the model used
        segment_type: Type of segmentation used
        max_cache_size_gb: Maximum cache file size in GB
    """
    # Get file stats to track document directory state
    file_stats = {}
    for segment_path, _ in segments:
        if os.path.isfile(segment_path):
            file_stats[segment_path] = {
                'mtime': os.path.getmtime(segment_path),
                'size': os.path.getsize(segment_path)
            }
    
    # Convert embeddings to numpy for better compatibility with pickle
    embedding_data = {
        'embeddings': embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings,
        'segments': segments,
        'timestamp': datetime.now().isoformat(),
        'doc_dir': doc_dir,
        'file_stats': file_stats,
        'model_name': model_name,
        'segment_type': segment_type,
    }
    
    # Estimate size before saving
    import sys
    import pickle as pkl
    
    # Create a temporary file to estimate size
    temp_data = pkl.dumps(embedding_data)
    estimated_size_gb = sys.getsizeof(temp_data) / (1024**3)
    
    if estimated_size_gb > max_cache_size_gb:
        print(f"Warning: Estimated cache size ({estimated_size_gb:.2f} GB) exceeds maximum ({max_cache_size_gb} GB)")
        print(f"Consider using chunked processing with smaller batches")
        
        # Ask for confirmation if very large
        if estimated_size_gb > 2 * max_cache_size_gb:
            confirmation = input(f"Cache file will be very large ({estimated_size_gb:.2f} GB). Continue? (y/n): ")
            if confirmation.lower() != 'y':
                print("Cache saving aborted by user")
                return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(embedding_data, f)
    
    # Get actual file size
    actual_size_mb = os.path.getsize(file_path) / (1024**2)
    print(f"Embeddings saved to {file_path} (Size: {actual_size_mb:.2f} MB)")

def load_embeddings(file_path: str) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    """
    Load embeddings and segments from a pickle file
    
    Args:
        file_path: Path to load the embeddings from
        
    Returns:
        Tuple of (embeddings, segments)
    """
    with open(file_path, 'rb') as f:
        embedding_data = pickle.load(f)
    
    # Convert numpy arrays back to torch tensors
    embeddings = torch.tensor(embedding_data['embeddings'])
    segments = embedding_data['segments']
    
    # Log when the embeddings were created
    saved_time = datetime.fromisoformat(embedding_data['timestamp'])
    print(f"Loaded embeddings from {file_path} (created on {saved_time})")
    
    return embeddings, segments

def is_cache_valid(cache_path: str, doc_dir: str, model_name: str, segment_type: str) -> bool:
    """
    Check if embedding cache is valid and up-to-date
    
    Args:
        cache_path: Path to the embedding cache file
        doc_dir: Path to the document directory
        model_name: Name of the model being used
        segment_type: Type of segmentation being used
        
    Returns:
        True if cache is valid, False otherwise
    """
    print(f"\nChecking cache validity: {cache_path}")
    
    # Simplified cache validation: only check if file exists
    if not os.path.isfile(cache_path):
        print(f"Cache file does not exist: {cache_path}")
        return False
    
    # Output cache file size and modification time
    cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path)).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Cache file size: {cache_size_mb:.2f} MB")
    print(f"Cache file modification time: {cache_mtime}")
    
    try:
        # Try to open the cache file, verify its structural integrity
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Output basic cache information
        cache_timestamp = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
        print(f"Cache creation time: {cache_timestamp}")
        print(f"Cache is valid, will use existing cache file")
        return True
    except Exception as e:
        print(f"Error checking cache validity: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_document_set_in_chunks(directory: str, segment_type: str, max_files: int = None, 
                             num_workers: int = NUM_WORKERS, max_batch_files: int = MAX_BATCH_FILES,
                             model: SentenceTransformer = None, batch_size: int = BATCH_SIZE) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    """
    Process a large document set in chunks to manage memory usage
    
    Args:
        directory: Path to the document set
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        max_files: Maximum number of files to process in total
        num_workers: Number of worker processes
        max_batch_files: Maximum number of files to process in each batch
        model: SentenceTransformer model for generating embeddings
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (embeddings, segments)
    """
    markdown_files = find_markdown_files(directory, max_files)
    total_files = len(markdown_files)
    
    print(f"Processing {total_files} markdown files in chunks of {max_batch_files}...")
    
    all_embeddings = []
    all_segments = []
    
    # Process files in batches
    for i in range(0, total_files, max_batch_files):
        batch_files = markdown_files[i:i+max_batch_files]
        print(f"Processing batch {i//max_batch_files + 1}/{(total_files + max_batch_files - 1)//max_batch_files}: {len(batch_files)} files")
        
        # Process current batch
        batch_segments = []
        if len(batch_files) < 10 or num_workers <= 1:
            for file_path in tqdm(batch_files, desc="Processing files"):
                try:
                    segments = extract_segments(file_path, segment_type)
                    batch_segments.extend(segments)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    print(f"Skipping file {file_path}")
        else:
            # For larger sets, use parallel processing
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.starmap(
                        extract_segments_safe, 
                        [(file_path, segment_type) for file_path in batch_files]
                    ),
                    total=len(batch_files),
                    desc="Processing files in parallel"
                ))
                for result in results:
                    batch_segments.extend(result)
        
        # Generate embeddings for the current batch
        if batch_segments:
            print(f"Generating embeddings for {len(batch_segments)} segments in current batch")
            batch_embeddings, batch_segments = compute_embeddings(model, batch_segments, batch_size)
            
            # Add to overall results
            if len(all_embeddings) == 0:
                all_embeddings = batch_embeddings
            else:
                # Convert to CPU tensors for concatenation if needed
                if isinstance(all_embeddings, torch.Tensor) and all_embeddings.is_cuda:
                    all_embeddings = all_embeddings.cpu()
                if isinstance(batch_embeddings, torch.Tensor) and batch_embeddings.is_cuda:
                    batch_embeddings = batch_embeddings.cpu()
                
                all_embeddings = torch.cat([all_embeddings, batch_embeddings], dim=0)
            
            all_segments.extend(batch_segments)
            
            # Free up memory
            del batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Processed {len(all_segments)} segments so far")
    
    print(f"Finished processing {total_files} files with {len(all_segments)} total segments")
    return all_embeddings, all_segments

def perform_second_stage_verification(similar_pairs: List[Tuple[Tuple[str, str], Tuple[str, str], float]], 
                                    second_model_name: str = SECOND_MODEL_NAME,
                                    mpnet_threshold: float = MPNET_SIMILARITY_THRESHOLD,
                                    batch_size: int = BATCH_SIZE,
                                    use_gpu: bool = USE_GPU) -> List[Tuple[Tuple[str, str], Tuple[str, str], float, float]]:
    """
    Perform second-stage verification on similar pairs using a more accurate model
    
    Args:
        similar_pairs: List of (segment1, segment2, similarity) tuples from first-stage filtering
        second_model_name: Name of the model to use for second-stage verification
        mpnet_threshold: Threshold for second-stage verification
        batch_size: Batch size for encoding
        use_gpu: Whether to use GPU for embedding generation
        
    Returns:
        List of (segment1, segment2, first_similarity, second_similarity) tuples that pass both thresholds
    """
    if not similar_pairs:
        return []
    
    # Separate short sentences from regular pairs
    short_sentence_pairs = []
    regular_pairs = []
    
    for pair in similar_pairs:
        segment1_text = pair[0][1]
        segment2_text = pair[1][1]
        
        # Check if either segment is a short sentence
        if is_short_sentence(segment1_text) or is_short_sentence(segment2_text):
            # For short sentences, add to results with a placeholder second similarity score (set to first similarity)
            short_sentence_pairs.append((pair[0], pair[1], pair[2], pair[2]))  # Use first similarity as placeholder
        else:
            regular_pairs.append(pair)
    
    print(f"\nPerforming second-stage verification with {second_model_name} model...")
    print(f"Verifying {len(regular_pairs)} regular candidate pairs with threshold {mpnet_threshold}")
    print(f"Skipping second-stage verification for {len(short_sentence_pairs)} short sentence pairs")
    
    if not regular_pairs:
        return short_sentence_pairs
        
    # Load the second model
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    second_model = SentenceTransformer(second_model_name, device=device)
    
    # Prepare text pairs for verification
    texts1 = [pair[0][1] for pair in regular_pairs]  # Extract text from segment1
    texts2 = [pair[1][1] for pair in regular_pairs]  # Extract text from segment2
    
    # Process in batches to avoid memory issues
    verified_pairs = []
    batch_count = (len(texts1) + batch_size - 1) // batch_size
    
    for i in tqdm(range(batch_count), desc="Verifying similarity batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts1))
        
        batch_texts1 = texts1[start_idx:end_idx]
        batch_texts2 = texts2[start_idx:end_idx]
        
        # Generate embeddings for both sets of texts
        emb1 = second_model.encode(batch_texts1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = second_model.encode(batch_texts2, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate cosine similarities
        mpnet_scores = util.cos_sim(emb1, emb2)
        
        # Check which pairs pass the second threshold
        for j in range(len(batch_texts1)):
            pair_idx = start_idx + j
            mpnet_sim = mpnet_scores[j][j].item()  # Diagonal element is the similarity between corresponding texts
            
            # Keep track of both similarity scores
            if mpnet_sim >= mpnet_threshold:
                verified_pairs.append((
                    regular_pairs[pair_idx][0],  # segment1
                    regular_pairs[pair_idx][1],  # segment2
                    regular_pairs[pair_idx][2],  # first-stage similarity
                    mpnet_sim                     # second-stage similarity
                ))
        
        # Clean up GPU memory
        del emb1, emb2, mpnet_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine verified regular pairs with short sentence pairs
    combined_pairs = verified_pairs + short_sentence_pairs
    
    print(f"Second-stage verification complete.")
    print(f"Retained {len(verified_pairs)} out of {len(regular_pairs)} regular pairs.")
    print(f"Total pairs: {len(combined_pairs)} (including {len(short_sentence_pairs)} short sentence pairs)")
    
    # Sort by second-stage similarity
    combined_pairs.sort(key=lambda x: x[3], reverse=True)
    
    return combined_pairs

def compare_document_sets(doc_set_1: str, doc_set_2: str, model_name: str = MODEL_NAME, 
                         second_model_name: str = SECOND_MODEL_NAME,
                         threshold: float = SIMILARITY_THRESHOLD, 
                         mpnet_threshold: float = MPNET_SIMILARITY_THRESHOLD,
                         segment_type: str = SEGMENT_TYPE,
                         batch_size: int = BATCH_SIZE, max_files: int = None, 
                         num_workers: int = NUM_WORKERS, use_gpu: bool = USE_GPU,
                         embedding_cache_1: str = None, embedding_cache_2: str = None,
                         use_cache: bool = USE_CACHE, chunked_processing: bool = CHUNKED_PROCESSING,
                         max_batch_files: int = MAX_BATCH_FILES, max_cache_size_gb: float = MAX_CACHE_SIZE_GB,
                         bert_weight: float = BERT_WEIGHT, jaccard_weight: float = JACCARD_WEIGHT) -> Dict:

    start_time = time.time()
    
    print(f"Loading SentenceTransformer model for first-stage filtering: {model_name}")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Using device: {device}")
    print(f"Using hybrid similarity with BERT weight: {bert_weight}, Jaccard weight: {jaccard_weight}")
    print(f"First-stage threshold: {threshold}, Second-stage threshold: {mpnet_threshold}")
    
    # Process document set 1
    if use_cache and embedding_cache_1 and is_cache_valid(embedding_cache_1, doc_set_1, model_name, segment_type):
        # Load embeddings from cache
        print(f"Loading cached embeddings for document set 1 from {embedding_cache_1}")
        embeddings1, segments1 = load_embeddings(embedding_cache_1)
    else:
        # Process documents and generate embeddings
        print(f"Processing document set 1: {doc_set_1}")
        
        if chunked_processing:
            print(f"Using chunked processing for document set 1 (batch size: {max_batch_files} files)")
            embeddings1, segments1 = process_document_set_in_chunks(
                doc_set_1, segment_type, max_files, num_workers, max_batch_files, model, batch_size
            )
        else:
            segments1 = process_document_set(doc_set_1, segment_type, max_files, num_workers)
            print(f"Found {len(segments1)} segments in document set 1")
            
            print("Computing embeddings for document set 1")
            embeddings1, segments1 = compute_embeddings(model, segments1, batch_size)
        
        # Save embeddings if cache path is provided
        if embedding_cache_1:
            print(f"Saving embeddings for document set 1 to {embedding_cache_1}")
            save_embeddings(embedding_cache_1, embeddings1, segments1, doc_set_1, model_name, segment_type, max_cache_size_gb)
    
    # Process document set 2
    if use_cache and embedding_cache_2 and is_cache_valid(embedding_cache_2, doc_set_2, model_name, segment_type):
        # Load embeddings from cache
        print(f"Loading cached embeddings for document set 2 from {embedding_cache_2}")
        embeddings2, segments2 = load_embeddings(embedding_cache_2)
    else:
        # Process documents and generate embeddings
        print(f"Processing document set 2: {doc_set_2}")
        
        if chunked_processing:
            print(f"Using chunked processing for document set 2 (batch size: {max_batch_files} files)")
            embeddings2, segments2 = process_document_set_in_chunks(
                doc_set_2, segment_type, max_files, num_workers, max_batch_files, model, batch_size
            )
        else:
            segments2 = process_document_set(doc_set_2, segment_type, max_files, num_workers)
            print(f"Found {len(segments2)} segments in document set 2")
            
            print("Computing embeddings for document set 2")
            embeddings2, segments2 = compute_embeddings(model, segments2, batch_size)
        
        # Save embeddings if cache path is provided
        if embedding_cache_2:
            print(f"Saving embeddings for document set 2 to {embedding_cache_2}")
            save_embeddings(embedding_cache_2, embeddings2, segments2, doc_set_2, model_name, segment_type, max_cache_size_gb)

    # First-stage filtering: Find similar segments using parallel processing
    print(f"First-stage filtering: Finding similar segments with threshold {threshold} using parallel processing and hybrid similarity")
    similar_pairs = find_similar_segments_parallel(
        segments1, embeddings1, segments2, embeddings2, threshold, num_workers, bert_weight, jaccard_weight
    )
    
    print(f"\nFirst-stage filtering complete. Found {len(similar_pairs)} candidate similar segment pairs")
    
    # Second-stage verification: Verify with more accurate model
    verified_pairs = perform_second_stage_verification(
        similar_pairs, second_model_name, mpnet_threshold, batch_size, use_gpu
    )
    
    # Group by file pairs
    file_pairs = {}
    for seg1, seg2, first_sim, second_sim in verified_pairs:
        file_pair = (seg1[0], seg2[0])
        if file_pair not in file_pairs:
            file_pairs[file_pair] = []
        file_pairs[file_pair].append((seg1[1], seg2[1], first_sim, second_sim))
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return {
        "total_similar_pairs": len(verified_pairs),
        "file_pairs": file_pairs,
        "processing_time": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def extract_title_from_markdown(file_path: str) -> str:
    """
    Extract the title from a markdown file by finding the first heading
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        The title of the document or the filename if title cannot be found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Find the first line that starts with # (ignoring whitespace)
                if line.strip().startswith('#'):
                    # Extract the title by removing the # and whitespace
                    return line.strip().lstrip('#').strip()
        
        # If no heading found, return the filename without extension
        return os.path.splitext(os.path.basename(file_path))[0]
    except Exception as e:
        print(f"Error extracting title from {file_path}: {e}")
        return os.path.basename(file_path)

def load_ignore_patterns(ignore_file: str = IGNORE_FILE) -> List[str]:
    """
    Load text patterns to ignore from a file
    
    Args:
        ignore_file: Path to the file containing patterns to ignore
        
    Returns:
        List of patterns to ignore
    """
    if not os.path.exists(ignore_file):
        print(f"Warning: Ignore file {ignore_file} not found. No patterns will be ignored.")
        return []
    
    try:
        with open(ignore_file, 'r', encoding='utf-8') as f:
            patterns = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(patterns)} patterns to ignore from {ignore_file}")
        return patterns
    except Exception as e:
        print(f"Error loading ignore patterns from {ignore_file}: {e}")
        return []

def should_ignore_text(text: str, ignore_patterns: List[str]) -> bool:

    # Normalize the text for comparison
    normalized_text = text.strip()
    
    # Check exact matches
    if normalized_text in ignore_patterns:
        return True
    
    return False

def export_results_to_excel(results: Dict, ignore_patterns: List[str] = None, excel_output_file: str = EXCEL_OUTPUT_FILE):

    # Split into regular and short sentences
    regular_pairs = []
    short_pairs = []
    
    for (file1, file2), pairs in results['file_pairs'].items():
        for text1, text2, first_sim, second_sim in pairs:
            # Skip if either segment matches an ignore pattern
            if ignore_patterns and (should_ignore_text(text1, ignore_patterns) or should_ignore_text(text2, ignore_patterns)):
                continue
                
            row_data = {
                'File1': os.path.abspath(file1),
                'File2': os.path.abspath(file2),
                'Text1': text1,
                'Text2': text2,
                'First Similarity': first_sim,
                'MPNet Similarity': second_sim
            }
            
            short_row_data = {
                'File1': os.path.abspath(file1),
                'File2': os.path.abspath(file2),
                'Text1': text1,
                'Text2': text2,
                'Similarity': first_sim  # For short sentences, we only show the first similarity score
            }
            
            if is_short_sentence(text1) or is_short_sentence(text2):
                short_pairs.append(short_row_data)
            else:
                regular_pairs.append(row_data)
    
    # Create DataFrames
    df_regular = pd.DataFrame(regular_pairs)
    df_short = pd.DataFrame(short_pairs)
    
    # Create Excel writer
    with pd.ExcelWriter(excel_output_file, engine='openpyxl') as writer:
        df_regular.to_excel(writer, sheet_name='Regular Pairs', index=False)
        df_short.to_excel(writer, sheet_name='Short Sentences', index=False)
    
    print(f"\nResults exported to Excel file: {excel_output_file}")
    print(f"  - Regular pairs: {len(regular_pairs)}")
    print(f"  - Short sentence pairs: {len(short_pairs)}")

def write_similarity_results(results: Dict, output_file: str, short_sentences_output_file: str = SHORT_SENTENCES_OUTPUT_FILE, 
                           excel_output_file: str = EXCEL_OUTPUT_FILE, ignore_file: str = IGNORE_FILE):
    # Load ignore patterns
    ignore_patterns = load_ignore_patterns(ignore_file)
    
    # Separate regular and short sentence pairs, filtering out ignored patterns
    regular_pairs = {}
    short_pairs = {}
    
    for (file1, file2), pairs in results['file_pairs'].items():
        regular_file_pairs = []
        short_file_pairs = []
        
        for seg1, seg2, first_sim, second_sim in pairs:
            # Skip if either segment matches an ignore pattern
            if ignore_patterns and (should_ignore_text(seg1, ignore_patterns) or should_ignore_text(seg2, ignore_patterns)):
                continue
                
            # Check if either segment is a short sentence
            if is_short_sentence(seg1) or is_short_sentence(seg2):
                short_file_pairs.append((seg1, seg2, first_sim, second_sim))
            else:
                regular_file_pairs.append((seg1, seg2, first_sim, second_sim))
        
        if regular_file_pairs:
            regular_pairs[(file1, file2)] = regular_file_pairs
        
        if short_file_pairs:
            short_pairs[(file1, file2)] = short_file_pairs
    
    # Write regular results
    regular_results = {
        "total_similar_pairs": sum(len(pairs) for pairs in regular_pairs.values()),
        "file_pairs": regular_pairs,
        "processing_time": results['processing_time'],
        "timestamp": results['timestamp']
    }
    
    # Write short sentences results
    short_results = {
        "total_similar_pairs": sum(len(pairs) for pairs in short_pairs.values()),
        "file_pairs": short_pairs,
        "processing_time": results['processing_time'],
        "timestamp": results['timestamp']
    }
    
    # Write regular results file
    _write_results_file(regular_results, output_file, "Regular")
    
    # Write short sentences results file
    _write_results_file(short_results, short_sentences_output_file, "Short Sentences")
    
    # Export to Excel
    export_results_to_excel(results, ignore_patterns, excel_output_file)
    
    print(f"\nRegular results written to {output_file}")
    print(f"Short sentences results written to {short_sentences_output_file}")

def _write_results_file(results: Dict, output_file: str, results_type: str):

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Document Similarity Check Results ({results_type})\n\n")
        f.write(f"**Timestamp:** {results['timestamp']}  \n\n")
        f.write(f"**Processing time:** {results['processing_time']:.2f} seconds\n\n")
        f.write(f"**Doc set 1:** {DOC_SET_1}\n")
        f.write(f"**Doc set 2:** {DOC_SET_2}\n\n")
        f.write(f"**Summary**: Found {results['total_similar_pairs']} similar segment pairs\n\n")
        
        # Reorganize results by doc-set-1 files
        doc1_files = {}
        for (file1, file2), pairs in results['file_pairs'].items():
            if file1 not in doc1_files:
                doc1_files[file1] = []
            doc1_files[file1].append((file2, pairs))
        
        # Write results grouped by doc-set-1 files
        for file1, comparisons in doc1_files.items():
            # Convert to relative paths
            rel_file1 = os.path.relpath(file1) if os.path.isabs(file1) else file1
            
            # Extract document title
            doc1_title = extract_title_from_markdown(file1)
            
            # Write heading for doc-set-1 file
            f.write(f"## Similar content in [{doc1_title}]({rel_file1})\n\n")
            
            total_segments = sum(len(pairs) for _, pairs in comparisons)
            f.write(f"**Found {total_segments} similar segments across {len(comparisons)} comparison(s)**\n\n")
            
            pair_counter = 1
            
            # Process each comparison
            for file2, pairs in comparisons:
                rel_file2 = os.path.relpath(file2) if os.path.isabs(file2) else file2
                
                f.write(f"Compared with: [`{rel_file2}`]({rel_file2})\n\n")
                
                for text1, text2, first_sim, second_sim in pairs:
                    # Check for binary or corrupted data
                    if not is_valid_text(text1) or not is_valid_text(text2):
                        print(f"Warning: Skipping pair {pair_counter} due to invalid text content")
                        continue
                    
                    # Check if this is a short sentence pair
                    is_short = is_short_sentence(text1) or is_short_sentence(text2)
                    
                    if is_short and results_type == "Short Sentences":
                        f.write(f"#### Pair {pair_counter} (Similarity: {first_sim:.4f})\n\n")
                    elif not is_short and results_type == "Regular":
                        f.write(f"#### Pair {pair_counter} (First similarity: {first_sim:.4f}, MPNet similarity: {second_sim:.4f})\n\n")
                    else:
                        # Skip if the pair doesn't match the results type
                        continue
                    
                    # Find line numbers for text1 in file1
                    line_num1 = find_line_number(file1, text1)
                    # Create a markdown link with line number in the URL
                    f.write(f"**[{rel_file1}: line {line_num1}]({rel_file1}#L{line_num1})**\n\n```\n{text1}\n```\n\n")
                    
                    # Find line numbers for text2 in file2
                    line_num2 = find_line_number(file2, text2)
                    # Create a markdown link with line number in the URL
                    f.write(f"**[{rel_file2}: line {line_num2}]({rel_file2}#L{line_num2})**\n\n```\n{text2}\n```\n\n")
                    
                    pair_counter += 1
                
                f.write("---\n\n")

def find_line_number(file_path: str, text_segment: str) -> int:

    # Try multiple encodings to handle various file formats
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                
            # Find the starting position of the text segment in the file
            start_pos = content.find(text_segment)
            
            if start_pos == -1:
                # Try to find a close match by removing whitespace
                clean_segment = ' '.join(text_segment.split())
                clean_content = ' '.join(content.split())
                start_pos = clean_content.find(clean_segment)
                
                if start_pos == -1:
                    # If still no match, try the next encoding or continue with default
                    continue
                    
                # If we found a match in the clean content, we need to map it back to the original
                # This is an approximation since whitespace has been normalized
                approx_pos = 0
                for i, c in enumerate(clean_content):
                    if i == start_pos:
                        break
                    if c != ' ':
                        approx_pos += 1
                start_pos = min(approx_pos, len(content) - 1)
            
            # Count the number of newlines before the starting position
            line_num = content[:start_pos].count('\n') + 1
            #print(f"Found text segment at line {line_num} in {file_path} using {encoding} encoding")
            return line_num
            
        except UnicodeDecodeError:
            # Try next encoding
            continue
        except Exception as e:
            print(f"Error finding line number in {file_path} with {encoding} encoding: {e}")
            # Try next encoding
            continue
    
    # If all encodings failed, return default
    # print(f"Could not find line number for text segment in {file_path} after trying multiple encodings")
    return 1

def create_embedding(text: str, model_name: str = MODEL_NAME) -> List[float]:

    if not text.strip():
        print("Warning: Empty text provided for embedding, returning zero vector")
        # Return zero vector with appropriate dimensions for the model
        return [0.0] * 384  # Common dimension for SentenceTransformer models
    
    try:
        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        
        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return a zero vector with a default dimension
        return [0.0] * 384  # Common dimension for SentenceTransformer models

def is_valid_text(text: str) -> bool:

    if not isinstance(text, str):
        return False
        
    # Check length
    if len(text) > 10000:  # Arbitrarily large text is likely not valid content
        return False
    
    # Check for a reasonable ratio of printable characters
    printable_chars = set(string.printable)
    printable_count = sum(1 for c in text if c in printable_chars)
    
    # If more than 15% of characters are non-printable, consider it invalid
    if len(text) > 0 and printable_count / len(text) < 0.85:
        return False
        
    return True

def is_short_sentence(text: str, word_threshold: int = 8) -> bool:

    # Remove punctuation and count words
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
    word_count = len(cleaned_text.split())
    
    return word_count < word_threshold

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Use command line arguments if provided, otherwise use global variables
    doc_set_1 = args.doc_set_1
    doc_set_2 = args.doc_set_2
    output_file = args.output
    short_sentences_output_file = args.short_output
    excel_output_file = args.excel_output
    ignore_file = args.ignore_file
    embedding_cache_1 = args.embedding_cache_1
    embedding_cache_2 = args.embedding_cache_2
    bert_weight = args.bert_weight
    jaccard_weight = args.jaccard_weight
    
    # Handle cache settings
    if args.no_cache:
        use_cache = False  # Command line flag takes precedence 
    else:
        use_cache = USE_CACHE  # Use the global setting from the top of the file
    
    chunked_processing = args.chunked
    max_batch_files = args.max_batch_files
    max_cache_size_gb = args.max_cache_size
    
    # Check if both document set paths are provided
    if not doc_set_1 or not doc_set_2:
        print("Error: You must provide paths for both document sets.")
        print("Either set the DOC_SET_1 and DOC_SET_2 variables in the script,")
        print("or provide them as command line arguments with --doc-set-1 and --doc-set-2.")
        return
    
    print(f"Cache usage setting: {'Enabled' if use_cache else 'Disabled'}")
    print(f"Chunked processing: {'Enabled' if chunked_processing else 'Disabled'}")
    print(f"Hybrid similarity: BERT weight: {bert_weight}, Jaccard weight: {jaccard_weight}")
    print(f"Using two-stage filtering with thresholds: {args.threshold} (first) and {args.mpnet_threshold} (second)")
    
    results = compare_document_sets(
        doc_set_1=doc_set_1,
        doc_set_2=doc_set_2,
        model_name=args.model,
        second_model_name=args.second_model,
        threshold=args.threshold,
        mpnet_threshold=args.mpnet_threshold,
        segment_type=args.segment,
        batch_size=args.batch_size,
        max_files=args.max_files,
        num_workers=args.workers,
        use_gpu=args.use_gpu,
        embedding_cache_1=embedding_cache_1,
        embedding_cache_2=embedding_cache_2,
        use_cache=use_cache,
        chunked_processing=chunked_processing,
        max_batch_files=max_batch_files,
        max_cache_size_gb=max_cache_size_gb,
        bert_weight=bert_weight,
        jaccard_weight=jaccard_weight
    )

    # Write results to files with separation of short sentences and export to Excel
    write_similarity_results(results, output_file, short_sentences_output_file, excel_output_file, ignore_file)

if __name__ == "__main__":
    main()
