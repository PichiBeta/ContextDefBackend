import spacy 
import numpy as np
from collections import Counter



def basic_features(doc):
    sentences = list(doc.sents)
    words = [token for token in doc if token.is_alpha]
    
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(words),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "avg_word_length": sum(len(t.text) for t in words) / max(len(words), 1),
    }

def lexical_diversity(doc):
    words = [token.lemma_.lower() for token in doc if token.is_alpha]
    unique_words = set(words)
    
    return {
        "type_token_ratio": len(unique_words) / max(len(words), 1)
    }

def syntactic_complexity(doc):
    depths = []
    for sent in doc.sents:
        depths.append(max([len(list(token.ancestors)) for token in sent]))
    
    return {
        "avg_tree_depth": sum(depths) / max(len(depths), 1)
    }

def subordinate_clauses(doc):
    subordinators = ["mark", "advcl", "ccomp", "xcomp"]
    count = sum(1 for token in doc if token.dep_ in subordinators)
    
    return {
        "subordination_ratio": count / max(len(list(doc.sents)), 1)
    }


POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "CCONJ", "SCONJ"]

def pos_distribution(doc):
    tokens = [t for t in doc if not t.is_space]
    total = len(tokens) or 1
    counts = {tag: 0 for tag in POS_TAGS}
    for t in tokens:
        if t.pos_ in counts:
            counts[t.pos_] += 1
    return {f"pos_{tag}": counts[tag] / total for tag in POS_TAGS}

