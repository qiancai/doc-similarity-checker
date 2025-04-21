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
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

# Configuration variables - modify these as needed
DOC_SET_1 = "/Users/grcai/Documents/GitHub/doc-similarity-checker/for-test/doc-set-1"  # Path to the first document set
DOC_SET_2 = "/Users/grcai/Documents/GitHub/doc-similarity-checker/for-test/doc-set-2"  # Path to the second document set
MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model name
SIMILARITY_THRESHOLD = 0.8  # Similarity threshold (0-1)
SEGMENT_TYPE = "paragraph"  # Text segmentation method ('paragraph' or 'sentence')
BATCH_SIZE = 32  # Batch size for encoding

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check similarity between two document sets")
    parser.add_argument("--doc-set-1", type=str, default=DOC_SET_1, help="Path to the first document set")
    parser.add_argument("--doc-set-2", type=str, default=DOC_SET_2, help="Path to the second document set")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Similarity threshold (0-1)")
    parser.add_argument("--segment", choices=["paragraph", "sentence"], default=SEGMENT_TYPE, 
                       help="Text segmentation method")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for encoding")
    return parser.parse_args()

def find_markdown_files(directory: str) -> List[str]:
    """
    Find all markdown files in the given directory and its subdirectories
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        List of paths to markdown files
    """
    return glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)

def read_markdown_file(file_path: str) -> str:
    """
    Read content from a markdown file
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Content of the file as a string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

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
    Split text into sentences (simple implementation)
    
    Args:
        text: The text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting by common end punctuation
    # A more sophisticated approach would use a dedicated NLP library
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

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
    
    if segment_type == "paragraph":
        segments = split_into_paragraphs(text)
    else:  # sentence
        segments = split_into_sentences(text)
    
    # Filter out too short segments and code blocks
    filtered_segments = []
    in_code_block = False
    for segment in segments:
        if segment.startswith("```") and segment.endswith("```"):
            continue
        if segment.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if len(segment.split()) >= 5:  # Only keep segments with at least 5 words
            filtered_segments.append(segment)
    
    return [(file_path, segment) for segment in filtered_segments]

def process_document_set(directory: str, segment_type: str) -> List[Tuple[str, str]]:
    """
    Process all markdown files in a document set
    
    Args:
        directory: Path to the document set
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        
    Returns:
        List of tuples containing (file_path, segment_text)
    """
    markdown_files = find_markdown_files(directory)
    all_segments = []
    
    for file_path in markdown_files:
        segments = extract_segments(file_path, segment_type)
        all_segments.extend(segments)
    
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
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
    return embeddings, segments

def find_similar_segments(segments1: List[Tuple[str, str]], embeddings1: torch.Tensor,
                         segments2: List[Tuple[str, str]], embeddings2: torch.Tensor, 
                         threshold: float) -> List[Tuple[Tuple[str, str], Tuple[str, str], float]]:
    """
    Find similar segments between two document sets
    
    Args:
        segments1: List of (file_path, segment_text) tuples from the first document set
        embeddings1: Embeddings for segments1
        segments2: List of (file_path, segment_text) tuples from the second document set
        embeddings2: Embeddings for segments2
        threshold: Similarity threshold
        
    Returns:
        List of (segment1, segment2, similarity_score) tuples
    """
    # Compute cosine similarity
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    # Find pairs with similarity scores above threshold
    similar_pairs = []
    for i in range(len(segments1)):
        for j in range(len(segments2)):
            if cosine_scores[i][j] >= threshold:
                similar_pairs.append((segments1[i], segments2[j], cosine_scores[i][j].item()))
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs

def compare_document_sets(doc_set_1: str, doc_set_2: str, model_name: str = MODEL_NAME, 
                         threshold: float = SIMILARITY_THRESHOLD, segment_type: str = SEGMENT_TYPE,
                         batch_size: int = BATCH_SIZE) -> Dict:
    """
    Compare two document sets and find similar content
    
    Args:
        doc_set_1: Path to the first document set
        doc_set_2: Path to the second document set
        model_name: SentenceTransformer model name
        threshold: Similarity threshold (0-1)
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary with similarity results
    """
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Processing document set 1: {doc_set_1}")
    segments1 = process_document_set(doc_set_1, segment_type)
    print(f"Found {len(segments1)} segments in document set 1")
    
    print(f"Processing document set 2: {doc_set_2}")
    segments2 = process_document_set(doc_set_2, segment_type)
    print(f"Found {len(segments2)} segments in document set 2")
    
    print("Computing embeddings for document set 1")
    embeddings1, segments1 = compute_embeddings(model, segments1, batch_size)
    
    print("Computing embeddings for document set 2")
    embeddings2, segments2 = compute_embeddings(model, segments2, batch_size)
    
    print(f"Finding similar segments with threshold {threshold}")
    similar_pairs = find_similar_segments(segments1, embeddings1, segments2, embeddings2, threshold)
    
    print(f"\nFound {len(similar_pairs)} similar segment pairs\n")
    
    # Group by file pairs
    file_pairs = {}
    for seg1, seg2, score in similar_pairs:
        file_pair = (seg1[0], seg2[0])
        if file_pair not in file_pairs:
            file_pairs[file_pair] = []
        file_pairs[file_pair].append((seg1[1], seg2[1], score))
    
    return {
        "total_similar_pairs": len(similar_pairs),
        "file_pairs": file_pairs
    }

def print_similarity_results(results: Dict):
    """
    Print similarity results in a readable format
    
    Args:
        results: Dictionary with similarity results
    """
    print(f"\nFound {results['total_similar_pairs']} similar segment pairs\n")
    
    # Print results grouped by file pairs
    for (file1, file2), pairs in results['file_pairs'].items():
        print(f"Similar content between files:")
        print(f"  File 1: {file1}")
        print(f"  File 2: {file2}")
        print(f"  Found {len(pairs)} similar segments\n")
        
        for i, (text1, text2, score) in enumerate(pairs, 1):
            print(f"  Pair {i} (similarity: {score:.4f}):")
            print(f"    Doc 1: {text1[:150]}..." if len(text1) > 150 else f"    Doc 1: {text1}")
            print(f"    Doc 2: {text2[:150]}..." if len(text2) > 150 else f"    Doc 2: {text2}")
            print()
        print("-" * 80)

def main():
    args = parse_args()
    
    # Use command line arguments if provided, otherwise use global variables
    doc_set_1 = args.doc_set_1
    doc_set_2 = args.doc_set_2
    
    # Check if both document set paths are provided
    if not doc_set_1 or not doc_set_2:
        print("Error: You must provide paths for both document sets.")
        print("Either set the DOC_SET_1 and DOC_SET_2 variables in the script,")
        print("or provide them as command line arguments with --doc-set-1 and --doc-set-2.")
        return
    
    results = compare_document_sets(
        doc_set_1=doc_set_1,
        doc_set_2=doc_set_2,
        model_name=args.model,
        threshold=args.threshold,
        segment_type=args.segment,
        batch_size=args.batch_size
    )

    print_similarity_results(results)

if __name__ == "__main__":
    main()
