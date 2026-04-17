#!/usr/bin/env python3
"""
Word Error Rate Calculator CLI
Supports WER, CER, BLEU, and other text comparison metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
from collections import Counter
import math


class TextMetrics:
    """Class containing various text comparison metrics."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text for better comparison.
        
        Args:
            text: Input text to clean
        
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove BOM (Byte Order Mark) if present
        if text.startswith('\ufeff'):
            text = text[1:]
        
        # Unicode normalization (NFC - canonical decomposition + canonical composition)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters and control characters (except newlines and tabs)
        text = ''.join(char for char in text if unicodedata.category(char) not in ['Cf', 'Cc'] or char in ['\n', '\t', ' '])
        
        # Normalize different types of whitespace to regular spaces
        # This includes non-breaking spaces, em spaces, en spaces, etc.
        text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]', ' ', text)
        
        # Normalize different types of quotes to standard ASCII quotes
        text = re.sub(r'[""''`´]', "'", text)  # Smart quotes to straight quotes
        text = re.sub(r'[«»‚„]', '"', text)   # Other quote types to double quotes
        
        # Normalize different types of dashes to standard hyphen
        text = re.sub(r'[–—−‒]', '-', text)   # Em dash, en dash, minus sign to hyphen
        
        # Normalize different types of apostrophes
        text = re.sub(r'[''`´]', "'", text)
        
        # Normalize ellipsis
        text = re.sub(r'…', '...', text)
        
        # Normalize line endings to space (treats newlines as word separators)
        text = re.sub(r'[\r\n]+', ' ', text)
        
        # Convert to lowercase for case-insensitive comparison
        cleaned = text.lower().strip()
        
        # Remove or normalize punctuation
        # Keep hyphens in compound words, but remove other punctuation
        cleaned = re.sub(r'[^\w\s\-]', ' ', cleaned)
        
        # Normalize multiple spaces/whitespace to single spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Final cleanup: remove extra spaces and strip
        cleaned = cleaned.strip()
        
        return cleaned
    
    @staticmethod
    def tokenize(text: str, clean_text: bool = True) -> List[str]:
        """
        Tokenize text into words with optional cleaning.
        
        Args:
            text: Input text to tokenize
            clean_text: Whether to clean the text before tokenization
        
        Returns:
            List of cleaned and normalized word tokens
        """
        if clean_text:
            text = TextMetrics.clean_text(text)
        
        # Split into words and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        return words
    
    @staticmethod
    def char_tokenize(text: str, clean_text: bool = True) -> List[str]:
        """
        Tokenize text into characters (excluding spaces) with optional cleaning.
        
        Args:
            text: Input text to tokenize
            clean_text: Whether to clean the text before tokenization
        
        Returns:
            List of characters (excluding spaces)
        """
        if clean_text:
            text = TextMetrics.clean_text(text)
        return [char for char in text.replace(' ', '')]
    
    @staticmethod
    def edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
        """Calculate edit distance and return (distance, substitutions, insertions, deletions)."""
        len_ref, len_hyp = len(ref), len(hyp)
        
        # Create DP table
        dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
        
        # Keep track of operations
        ops = [[('', 0, 0, 0)] * (len_hyp + 1) for _ in range(len_ref + 1)]
        
        # Initialize base cases
        for i in range(len_ref + 1):
            dp[i][0] = i
            ops[i][0] = ('del', 0, 0, i)
        for j in range(len_hyp + 1):
            dp[0][j] = j
            ops[0][j] = ('ins', 0, j, 0)
        
        # Fill DP table
        for i in range(1, len_ref + 1):
            for j in range(1, len_hyp + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = ops[i-1][j-1]
                else:
                    # Substitution
                    sub_cost = dp[i-1][j-1] + 1
                    # Insertion
                    ins_cost = dp[i][j-1] + 1
                    # Deletion
                    del_cost = dp[i-1][j] + 1
                    
                    min_cost = min(sub_cost, ins_cost, del_cost)
                    dp[i][j] = min_cost
                    
                    if min_cost == sub_cost:
                        prev_ops = ops[i-1][j-1]
                        ops[i][j] = (prev_ops[0], prev_ops[1] + 1, prev_ops[2], prev_ops[3])
                    elif min_cost == ins_cost:
                        prev_ops = ops[i][j-1]
                        ops[i][j] = (prev_ops[0], prev_ops[1], prev_ops[2] + 1, prev_ops[3])
                    else:  # deletion
                        prev_ops = ops[i-1][j]
                        ops[i][j] = (prev_ops[0], prev_ops[1], prev_ops[2], prev_ops[3] + 1)
        
        final_ops = ops[len_ref][len_hyp]
        return dp[len_ref][len_hyp], final_ops[1], final_ops[2], final_ops[3]
    
    @classmethod
    def word_error_rate(cls, reference: str, hypothesis: str, clean_text: bool = False) -> Dict[str, Any]:
        """Calculate Word Error Rate (WER)."""
        ref_words = cls.tokenize(reference, clean_text)
        hyp_words = cls.tokenize(hypothesis, clean_text)
        
        if len(ref_words) == 0:
            return {
                'wer': float('inf') if len(hyp_words) > 0 else 0.0,
                'substitutions': 0,
                'insertions': len(hyp_words),
                'deletions': 0,
                'total_words': 0,
                'reference_length': 0,
                'hypothesis_length': len(hyp_words)
            }
        
        distance, subs, ins, dels = cls.edit_distance(ref_words, hyp_words)
        wer = distance / len(ref_words)
        
        return {
            'wer': wer,
            'substitutions': subs,
            'insertions': ins,
            'deletions': dels,
            'total_words': len(ref_words),
            'reference_length': len(ref_words),
            'hypothesis_length': len(hyp_words)
        }
    
    @classmethod
    def character_error_rate(cls, reference: str, hypothesis: str, clean_text: bool = False) -> Dict[str, Any]:
        """Calculate Character Error Rate (CER)."""
        ref_chars = cls.char_tokenize(reference, clean_text)
        hyp_chars = cls.char_tokenize(hypothesis, clean_text)
        
        if len(ref_chars) == 0:
            return {
                'cer': float('inf') if len(hyp_chars) > 0 else 0.0,
                'substitutions': 0,
                'insertions': len(hyp_chars),
                'deletions': 0,
                'total_chars': 0,
                'reference_length': 0,
                'hypothesis_length': len(hyp_chars)
            }
        
        distance, subs, ins, dels = cls.edit_distance(ref_chars, hyp_chars)
        cer = distance / len(ref_chars)
        
        return {
            'cer': cer,
            'substitutions': subs,
            'insertions': ins,
            'deletions': dels,
            'total_chars': len(ref_chars),
            'reference_length': len(ref_chars),
            'hypothesis_length': len(hyp_chars)
        }
    
    @classmethod
    def bleu_score(cls, reference: str, hypothesis: str, max_n: int = 4, clean_text: bool = False) -> Dict[str, Any]:
        """Calculate BLEU score."""
        ref_words = cls.tokenize(reference, clean_text)
        hyp_words = cls.tokenize(hypothesis, clean_text)
        
        if len(hyp_words) == 0:
            return {'bleu': 0.0, 'brevity_penalty': 0.0, 'precision_scores': [0.0] * max_n}
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = cls._get_ngrams(ref_words, n)
            hyp_ngrams = cls._get_ngrams(hyp_words, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = 0
            for ngram in hyp_ngrams:
                if ngram in ref_ngrams:
                    matches += min(hyp_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(hyp_ngrams.values())
            precisions.append(precision)
        
        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_sum = sum(math.log(p) for p in precisions)
            bleu = math.exp(log_sum / len(precisions))
        
        # Brevity penalty
        ref_len = len(ref_words)
        hyp_len = len(hyp_words)
        
        if hyp_len > ref_len:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - ref_len / hyp_len)
        
        final_bleu = bleu * brevity_penalty
        
        return {
            'bleu': final_bleu,
            'brevity_penalty': brevity_penalty,
            'precision_scores': precisions,
            'reference_length': ref_len,
            'hypothesis_length': hyp_len
        }
    
    @staticmethod
    def _get_ngrams(words: List[str], n: int) -> Counter:
        """Get n-grams from a list of words."""
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    @classmethod
    def jaccard_similarity(cls, reference: str, hypothesis: str, clean_text: bool = False) -> Dict[str, Any]:
        """Calculate Jaccard similarity coefficient."""
        ref_words = set(cls.tokenize(reference, clean_text))
        hyp_words = set(cls.tokenize(hypothesis, clean_text))
        
        intersection = ref_words.intersection(hyp_words)
        union = ref_words.union(hyp_words)
        
        if len(union) == 0:
            jaccard = 1.0  # Both sets are empty
        else:
            jaccard = len(intersection) / len(union)
        
        return {
            'jaccard': jaccard,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'reference_unique_words': len(ref_words),
            'hypothesis_unique_words': len(hyp_words)
        }


def read_file_content(filepath: str) -> str:
    """Read content from a file."""
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return path.read_text(encoding='utf-8').strip()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def calculate_wer_from_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> Dict[str, Any]:
    """Calculate WER from pre-tokenized word lists."""
    if len(ref_tokens) == 0:
        return {
            'wer': float('inf') if len(hyp_tokens) > 0 else 0.0,
            'substitutions': 0,
            'insertions': len(hyp_tokens),
            'deletions': 0,
            'total_words': 0,
            'reference_length': 0,
            'hypothesis_length': len(hyp_tokens)
        }
    
    distance, subs, ins, dels = TextMetrics.edit_distance(ref_tokens, hyp_tokens)
    wer = distance / len(ref_tokens)
    
    return {
        'wer': wer,
        'substitutions': subs,
        'insertions': ins,
        'deletions': dels,
        'total_words': len(ref_tokens),
        'reference_length': len(ref_tokens),
        'hypothesis_length': len(hyp_tokens)
    }


def calculate_cer_from_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> Dict[str, Any]:
    """Calculate CER from pre-tokenized character lists."""
    if len(ref_tokens) == 0:
        return {
            'cer': float('inf') if len(hyp_tokens) > 0 else 0.0,
            'substitutions': 0,
            'insertions': len(hyp_tokens),
            'deletions': 0,
            'total_chars': 0,
            'reference_length': 0,
            'hypothesis_length': len(hyp_tokens)
        }
    
    distance, subs, ins, dels = TextMetrics.edit_distance(ref_tokens, hyp_tokens)
    cer = distance / len(ref_tokens)
    
    return {
        'cer': cer,
        'substitutions': subs,
        'insertions': ins,
        'deletions': dels,
        'total_chars': len(ref_tokens),
        'reference_length': len(ref_tokens),
        'hypothesis_length': len(hyp_tokens)
    }


def format_results(metric_name: str, results: Dict[str, Any]) -> str:
    """Format results for display."""
    output = [f"\n=== {metric_name.upper()} Results ==="]
    
    if metric_name.lower() == 'wer':
        output.append(f"Word Error Rate: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
        output.append(f"Substitutions: {results['substitutions']}")
        output.append(f"Insertions: {results['insertions']}")
        output.append(f"Deletions: {results['deletions']}")
        output.append(f"Reference length: {results['reference_length']} words")
        output.append(f"Hypothesis length: {results['hypothesis_length']} words")
    
    elif metric_name.lower() == 'cer':
        output.append(f"Character Error Rate: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
        output.append(f"Substitutions: {results['substitutions']}")
        output.append(f"Insertions: {results['insertions']}")
        output.append(f"Deletions: {results['deletions']}")
        output.append(f"Reference length: {results['reference_length']} characters")
        output.append(f"Hypothesis length: {results['hypothesis_length']} characters")
    
    elif metric_name.lower() == 'bleu':
        output.append(f"BLEU Score: {results['bleu']:.4f}")
        output.append(f"Brevity Penalty: {results['brevity_penalty']:.4f}")
        for i, score in enumerate(results['precision_scores']):
            output.append(f"{i+1}-gram Precision: {score:.4f}")
        output.append(f"Reference length: {results['reference_length']} words")
        output.append(f"Hypothesis length: {results['hypothesis_length']} words")
    
    elif metric_name.lower() == 'jaccard':
        output.append(f"Jaccard Similarity: {results['jaccard']:.4f}")
        output.append(f"Intersection size: {results['intersection_size']}")
        output.append(f"Union size: {results['union_size']}")
        output.append(f"Reference unique words: {results['reference_unique_words']}")
        output.append(f"Hypothesis unique words: {results['hypothesis_unique_words']}")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate text comparison metrics (WER, CER, BLEU, Jaccard)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s reference.txt hypothesis.txt
  %(prog)s reference.txt hypothesis.txt --metric cer
  %(prog)s reference.txt hypothesis.txt --metric bleu
  %(prog)s reference.txt hypothesis.txt --metric all
  %(prog)s reference.txt hypothesis.txt --metric jaccard --verbose
  %(prog)s v2.txt v3.txt --verbose
  %(prog)s v2.txt v3.txt --clean --verbose
  %(prog)s v2.txt v3.txt --no-clean --verbose
"""
    )
    
    parser.add_argument('reference_file', 
                       help='Path to the reference (ground truth) text file')
    parser.add_argument('hypothesis_file',
                       help='Path to the hypothesis (predicted) text file')
    parser.add_argument('-m', '--metric', 
                       choices=['wer', 'cer', 'bleu', 'jaccard', 'all'],
                       default='wer',
                       help='Metric to calculate (default: wer)')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Show additional details and file contents')
    parser.add_argument('-c', '--clean',
                       action='store_true',
                       help='Enable text cleaning and normalization')
    parser.add_argument('--no-clean',
                       action='store_true',
                       help='Disable text cleaning (compare raw text)')
    
    args = parser.parse_args()
    
    # Determine cleaning options - default is cleaning enabled
    if args.no_clean:
        enable_cleaning = False
    elif args.clean:
        enable_cleaning = True
    else:
        # Default behavior: cleaning enabled
        enable_cleaning = True
    
    # Read files
    reference_text = read_file_content(args.reference_file)
    hypothesis_text = read_file_content(args.hypothesis_file)
    
    if args.verbose:
        print(f"Reference file: {args.reference_file}")
        print(f"Reference content (raw): '{reference_text[:200]}{'...' if len(reference_text) > 200 else ''}'")
        
        if enable_cleaning:
            cleaned_ref = TextMetrics.clean_text(reference_text)
            print(f"Reference content (cleaned): '{cleaned_ref[:200]}{'...' if len(cleaned_ref) > 200 else ''}'")
        
        print(f"\nHypothesis file: {args.hypothesis_file}")
        print(f"Hypothesis content (raw): '{hypothesis_text[:200]}{'...' if len(hypothesis_text) > 200 else ''}'")
        
        if enable_cleaning:
            cleaned_hyp = TextMetrics.clean_text(hypothesis_text)
            print(f"Hypothesis content (cleaned): '{cleaned_hyp[:200]}{'...' if len(cleaned_hyp) > 200 else ''}'")
        
        print(f"\nCleaning enabled: {enable_cleaning}")
        print("\n" + "="*60)
    
    # Calculate metrics
    metrics = TextMetrics()
    
    if args.metric == 'wer' or args.metric == 'all':
        wer_results = metrics.word_error_rate(reference_text, hypothesis_text, enable_cleaning)
        print(format_results('WER', wer_results))
    
    if args.metric == 'cer' or args.metric == 'all':
        cer_results = metrics.character_error_rate(reference_text, hypothesis_text, enable_cleaning)
        print(format_results('CER', cer_results))
    
    if args.metric == 'bleu' or args.metric == 'all':
        bleu_results = metrics.bleu_score(reference_text, hypothesis_text, 4, enable_cleaning)
        print(format_results('BLEU', bleu_results))
    
    if args.metric == 'jaccard' or args.metric == 'all':
        jaccard_results = metrics.jaccard_similarity(reference_text, hypothesis_text, enable_cleaning)
        print(format_results('Jaccard', jaccard_results))


if __name__ == "__main__":
    main()
