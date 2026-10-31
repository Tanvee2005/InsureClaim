"""Core NLP: preprocessing + NLI feature engineering (from NLP_Project.ipynb, Assignments 1-3)."""
import re, string
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity

for _p in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
           'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(_p, quiet=True)
    except Exception:
        pass

NEGATIONS = {"no", "not", "nor", "never", "none", "neither", "n't", "cannot", "without"}
STOPWORDS = set(stopwords.words('english')) - NEGATIONS
lemmatizer = WordNetLemmatizer()


def _wn_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    if tag.startswith('V'): return wordnet.VERB
    if tag.startswith('N'): return wordnet.NOUN
    if tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text):
    """Returns (sentences, lemmatised_tokens, cleaned_text)."""
    sentences = sent_tokenize(str(text))
    out = []
    for sent in sentences:
        toks = [t for t in word_tokenize(normalize_text(sent))
                if t not in STOPWORDS and t not in string.punctuation]
        out.extend(lemmatizer.lemmatize(t, _wn_pos(g)) for t, g in pos_tag(toks))
    return sentences, out, " ".join(out)


def clean(text):
    return preprocess(text)[2]


def sent_vec(text, w2v, dim=100):
    vs = [w2v.wv[w] for w in str(text).split() if w in w2v.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(dim)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def pair_features(c1, c2, w2v, pair_tfidf):
    """Feature matrix for lists of cleaned premise/hypothesis strings."""
    v1 = pair_tfidf.transform(c1); v2 = pair_tfidf.transform(c2)
    tf_cos = np.asarray(v1.multiply(v2).sum(axis=1)).ravel()
    rows = []
    for i, (a, b) in enumerate(zip(c1, c2)):
        wa, wb = set(a.split()), set(b.split())
        union = len(wa | wb) or 1
        inter = len(wa & wb)
        overlap = inter / union
        hyp_cov = inter / (len(wb) or 1)          # how much of the hypothesis is in the premise
        len_a, len_b = len(a.split()), len(b.split())
        neg = int((len(wa & NEGATIONS) > 0) != (len(wb & NEGATIONS) > 0))
        ea, eb = sent_vec(a, w2v), sent_vec(b, w2v)
        rows.append(np.concatenate([
            [overlap, hyp_cov, abs(len_a - len_b), len_a, len_b, neg, tf_cos[i], _cos(ea, eb)],
            np.abs(ea - eb), ea * eb]))
    return np.asarray(rows)

FEATURE_NAMES_BASE = ['word_overlap', 'hypothesis_coverage', 'length_diff', 'len_premise',
                      'len_hypothesis', 'negation_mismatch', 'tfidf_cosine', 'w2v_cosine']
