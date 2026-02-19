import spacy
from feature_funcs.feature_funcs import *

nlp = spacy.load("en_core_web_md")


def extract_features(text):
    doc = nlp(text)
    
    features = {}
    features.update(basic_features(doc))
    features.update(lexical_diversity(doc))
    features.update(syntactic_complexity(doc))
    features.update(subordinate_clauses(doc))
    features.update(pos_distribution(doc))
    
    return features