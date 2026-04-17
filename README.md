# WER Calculator CLI

A Python command-line tool for calculating Word Error Rate (WER) and other text comparison metrics with advanced text cleaning capabilities for real-world data.

## Features

### Core Metrics
- **Word Error Rate (WER)**: Measures word-level differences between reference and hypothesis texts
- **Character Error Rate (CER)**: Measures character-level differences
- **BLEU Score**: Bilingual Evaluation Understudy score for translation quality
- **Jaccard Similarity**: Measures similarity between word sets

### Text Cleaning
- **Smart Normalization**: Case normalization, punctuation handling, whitespace cleanup

## Installation

## Usage

### Basic Usage
```bash
# Calculate WER (default metric) with cleaning enabled
uv sync

uv run wercalc.py reference.txt hypothesis.txt
```

### Different Metrics
```bash
# Character Error Rate
uv run wercalc.py reference.txt hypothesis.txt --metric cer

# BLEU Score
uv run wercalc.py reference.txt hypothesis.txt --metric bleu

# Jaccard Similarity
uv run wercalc.py reference.txt hypothesis.txt --metric jaccard

# Calculate all metrics
uv run wercalc.py reference.txt hypothesis.txt --metric all
```

### Text Cleaning Options
```bash
# Default: cleaning and aggressive cleaning enabled
uv run wercalc.py reference.txt hypothesis.txt --verbose

# Disable all cleaning (compare raw text)
uv run wercalc.py reference.txt hypothesis.txt --no-clean
```

## Text Cleaning Features

### Automatic Normalization
- Case normalization (lowercase)
- Punctuation cleanup
- Whitespace normalization
- Special character handling

## Real-world Example

### Without Cleaning
```bash
uv run wercalc.py reference.txt hypothesis.txt --no-clean --metric all
```

### With Cleaning (Default)
```bash
uv run wercalc.py reference.txt hypothesis.txt --metric all
```

## Example

Run the tool:
```bash
uv run wercalc.py reference.txt hypothesis.txt --metric all --verbose
```

## Metrics Explained

### Word Error Rate (WER)
- Formula: `(S + D + I) / N`
- Where S = substitutions, D = deletions, I = insertions, N = total words in reference
- Lower is better (0 = perfect match)

### Character Error Rate (CER)
- Similar to WER but calculated at character level
- Useful for languages without clear word boundaries

### BLEU Score
- Measures n-gram overlap between reference and hypothesis
- Includes brevity penalty for shorter translations
- Higher is better (1.0 = perfect match)

### Jaccard Similarity
- Measures overlap of unique words between texts
- Formula: `|intersection| / |union|`
- Higher is better (1.0 = identical word sets)

## Requirements

- No external dependencies (uses only Python standard library)