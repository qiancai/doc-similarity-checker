# Document Similarity Checker

A tool for comparing two sets of markdown documents to identify similar content.

## Overview

This tool helps detect similarity between two sets of markdown documents using a two-stage hybrid approach:

1. **First-stage filtering**: Uses the "all-MiniLM-L6-v2" model to generate embeddings and combines BERT similarity with Jaccard similarity for initial filtering.
2. **Second-stage verification**: Uses the more accurate "all-mpnet-base-v2" model to recalculate similarity for candidate pairs.

## Features

- Semantic understanding of document content
- Two-stage similarity checking for accuracy
- Hybrid similarity calculation (BERT + Jaccard)
- Separate handling for short sentences
- Exports results in markdown and Excel formats
- Supports embedding caching for faster processing
- Chunked processing for large document sets
- Parallel processing for improved performance
- Configurable thresholds and parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/doc-similarity-checker.git
cd doc-similarity-checker

# Install required dependencies
pip install sentence-transformers torch numpy tqdm pandas openpyxl
```

## Usage

```bash
# Basic usage
python scripts/doc-similarity-checker.py --doc-set-1 /path/to/first/docs --doc-set-2 /path/to/second/docs

# Advanced usage with custom parameters
python scripts/doc-similarity-checker.py \
  --doc-set-1 /path/to/first/docs \
  --doc-set-2 /path/to/second/docs \
  --threshold 0.65 \
  --mpnet-threshold 0.85 \
  --segment sentence \
  --use-gpu
```

Alternatively, you can edit the parameters in the `doc-similarity-checker.py` script first, and then run the script directly.

```bash
python scripts/doc-similarity-checker.py
```

## Command line arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--doc-set-1` | Path to the first document set | From script config |
| `--doc-set-2` | Path to the second document set | From script config |
| `--model` | SentenceTransformer model for first-stage filtering | all-MiniLM-L6-v2 |
| `--second-model` | SentenceTransformer model for second-stage verification | all-mpnet-base-v2 |
| `--threshold` | First-stage similarity threshold (0-1) | 0.60 |
| `--mpnet-threshold` | Second-stage similarity threshold (0-1) | 0.80 |
| `--bert-weight` | Weight for BERT similarity in hybrid approach (0-1) | 0.7 |
| `--jaccard-weight` | Weight for Jaccard similarity in hybrid approach (0-1) | 0.3 |
| `--segment` | Text segmentation method (paragraph or sentence) | sentence |
| `--batch-size` | Batch size for encoding | 64 |
| `--output` | Output file for results | check_results.md |
| `--short-output` | Output file for short sentences results | check_results_short_sentences.md |
| `--excel-output` | Excel output file | similarity_results.xlsx |
| `--ignore-file` | File containing text patterns to ignore | ignore.txt |
| `--workers` | Number of worker processes | CPU count - 1 |
| `--max-files` | Maximum number of files to process per document set | None |
| `--use-gpu` | Use GPU for embedding generation | Auto-detect |
| `--no-cache` | Do not use embedding cache | False |
| `--chunked` | Enable chunked processing for large document sets | True |
| `--max-batch-files` | Maximum files to process in each batch | 50 |
| `--max-cache-size` | Maximum cache file size in GB | 2 |

## Output

The tool generates three output files:

- `check_results.md`: Markdown file with regular similarity results
- `check_results_short_sentences.md`: Markdown file with short sentence similarity results
- `similarity_results.xlsx`: Excel file with all results in separate sheets

## Ignoring text

Create an `ignore.txt` file with text to exclude from similarity checking:

```
This is a common text to ignore.
Another text pattern to ignore.
```

## Performance Considerations

- For large document sets, use chunked processing (`--chunked`)
- GPU acceleration is recommended for faster processing
- Embedding caching can significantly speed up repeated runs
- Adjust worker count based on your CPU capabilities
- Reduce batch size if you encounter memory issues

## License

[MIT License](LICENSE)