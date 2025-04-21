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
from sentence_transformers import SentenceTransformer, util
import torch

# Configuration variables - modify these as needed
DOC_SET_1 = "/Users/grcai/Documents/GitHub/doc-similarity-checker/for-test/doc-set-1"  # Path to the first document set
DOC_SET_2 = "/Users/grcai/Documents/GitHub/doc-similarity-checker/for-test/doc-set-2"  # Path to the second document set
MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model name
SIMILARITY_THRESHOLD = 0.8  # Similarity threshold (0-1)f
SEGMENT_TYPE = "sentence"  # Text segmentation method ('paragraph' or 'sentence')
BATCH_SIZE = 64  # Batch size for encoding
OUTPUT_FILE = "check_results.md"  # Output file for results (changed to .md)
USE_GPU = torch.cuda.is_available()  # Use GPU if available
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Number of worker processes for parallel processing

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
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file for results")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process per document set")
    parser.add_argument("--use-gpu", action="store_true", default=USE_GPU, help="Use GPU for embedding generation")
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
            segments = extract_segments(file_path, segment_type)
            all_segments.extend(segments)
    else:
        # For larger sets, use parallel processing
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.starmap(
                    extract_segments, 
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
    print("Computing cosine similarities...")
    start_time = time.time()
    
    # For large embeddings, process in chunks to avoid memory issues
    chunk_size = 1000  # Adjust based on available memory
    similar_pairs = []
    
    total_chunks = (len(embeddings1) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(0, len(embeddings1), chunk_size), total=total_chunks, desc="Comparing chunks"):
        # Get current chunk
        embeddings1_chunk = embeddings1[i:i+chunk_size]
        segments1_chunk = segments1[i:i+chunk_size]
        
        # Compute cosine similarity for current chunk
        cosine_scores = util.cos_sim(embeddings1_chunk, embeddings2)
        
        # Find pairs with similarity scores above threshold
        for chunk_idx, scores in enumerate(cosine_scores):
            orig_idx = i + chunk_idx
            for j, score in enumerate(scores):
                if score >= threshold:
                    similar_pairs.append((segments1_chunk[chunk_idx], segments2[j], score.item()))
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    elapsed_time = time.time() - start_time
    print(f"Found {len(similar_pairs)} similar pairs in {elapsed_time:.2f} seconds")
    
    return similar_pairs

def compare_document_sets(doc_set_1: str, doc_set_2: str, model_name: str = MODEL_NAME, 
                         threshold: float = SIMILARITY_THRESHOLD, segment_type: str = SEGMENT_TYPE,
                         batch_size: int = BATCH_SIZE, max_files: int = None, 
                         num_workers: int = NUM_WORKERS, use_gpu: bool = USE_GPU) -> Dict:
    """
    Compare two document sets and find similar content
    
    Args:
        doc_set_1: Path to the first document set
        doc_set_2: Path to the second document set
        model_name: SentenceTransformer model name
        threshold: Similarity threshold (0-1)
        segment_type: Type of segmentation ('paragraph' or 'sentence')
        batch_size: Batch size for encoding
        max_files: Maximum number of files to process per document set
        num_workers: Number of worker processes
        use_gpu: Whether to use GPU for embedding generation
        
    Returns:
        Dictionary with similarity results
    """
    start_time = time.time()
    
    print(f"Loading SentenceTransformer model: {model_name}")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    print(f"Using device: {device}")
    
    print(f"Processing document set 1: {doc_set_1}")
    segments1 = process_document_set(doc_set_1, segment_type, max_files, num_workers)
    print(f"Found {len(segments1)} segments in document set 1")
    
    print(f"Processing document set 2: {doc_set_2}")
    segments2 = process_document_set(doc_set_2, segment_type, max_files, num_workers)
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
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return {
        "total_similar_pairs": len(similar_pairs),
        "file_pairs": file_pairs,
        "processing_time": total_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def write_similarity_results(results: Dict, output_file: str):
    """
    Write similarity results to a Markdown file
    
    Args:
        results: Dictionary with similarity results
        output_file: Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Document Similarity Check Results\n\n")
        f.write(f"**Timestamp:** {results['timestamp']}  \n")
        f.write(f"**Processing time:** {results['processing_time']:.2f} seconds  \n\n")
        f.write(f"**Found {results['total_similar_pairs']} similar segment pairs**\n\n")
        
        # Write results grouped by file pairs
        for (file1, file2), pairs in results['file_pairs'].items():
            # Convert to relative paths if possible
            rel_file1 = os.path.relpath(file1) if os.path.isabs(file1) else file1
            rel_file2 = os.path.relpath(file2) if os.path.isabs(file2) else file2
            
            # Changed heading to focus on the file in set 1
            f.write(f"## Similar content in [`{rel_file1}`]({rel_file1})\n\n")
            f.write(f"**Compared with:** [`{rel_file2}`]({rel_file2})  \n")
            f.write(f"**Found {len(pairs)} similar segments**\n\n")
            
            for i, (text1, text2, score) in enumerate(pairs, 1):
                f.write(f"### Pair {i} (similarity: {score:.4f})\n\n")
                
                # Find line numbers for text1 in file1
                line_num1 = find_line_number(file1, text1)
                # Create a markdown link with line number in the URL
                f.write(f"**[{rel_file1}: line {line_num1}]({rel_file1}#L{line_num1})**\n\n```\n{text1}\n```\n\n")
                
                # Find line numbers for text2 in file2
                line_num2 = find_line_number(file2, text2)
                # Create a markdown link with line number in the URL
                f.write(f"**[{rel_file2}: line {line_num2}]({rel_file2}#L{line_num2})**\n\n```\n{text2}\n```\n\n")
            f.write("---\n\n")
    
    print(f"\nResults written to {output_file}")

def find_line_number(file_path: str, text_segment: str) -> int:
    """
    Find the line number where a text segment appears in a file
    
    Args:
        file_path: Path to the file
        text_segment: The text segment to find
        
    Returns:
        Line number (1-indexed) where the segment starts, or 1 if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the starting position of the text segment in the file
        start_pos = content.find(text_segment)
        
        if start_pos == -1:
            # If exact match not found, try to find a close match
            return 1
            
        # Count the number of newlines before the starting position
        line_num = content[:start_pos].count('\n') + 1
        return line_num
    except Exception as e:
        print(f"Error finding line number in {file_path}: {e}")
        return 1

def main():
    args = parse_args()
    
    # Use command line arguments if provided, otherwise use global variables
    doc_set_1 = args.doc_set_1
    doc_set_2 = args.doc_set_2
    output_file = args.output
    
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
        batch_size=args.batch_size,
        max_files=args.max_files,
        num_workers=args.workers,
        use_gpu=args.use_gpu
    )

    # Write results to file
    write_similarity_results(results, output_file)

if __name__ == "__main__":
    main()
