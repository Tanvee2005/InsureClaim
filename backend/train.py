"""Train all models. Usage: python backend/train.py   (≈3-6 min on CPU)"""
import json, time, joblib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from nlp import clean, pair_features, FEATURE_NAMES_BASE

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
D, A = ROOT / 'data', ROOT / 'artifacts'
A.mkdir(exist_ok=True)
LABELS = ['entailment', 'neutral', 'contradiction']
metrics = {}


def load_snli(name):
    df = pd.read_csv(D / 'snli' / f'{name}.csv').dropna()
    df = df[df.gold_label.isin(LABELS)].reset_index(drop=True)
    t = time.time()
    df['c1'] = df.sentence1.map(clean); df['c2'] = df.sentence2.map(clean)
    print(f'  preprocessed {name}: {len(df)} rows in {time.time()-t:.0f}s')
    return df

print('[1/4] Preprocessing SNLI + policy clauses')
tr, dv, te = load_snli('train_40k'), load_snli('dev_4k'), load_snli('test_4k')
pol = pd.read_csv(D / 'policy_clauses.csv'); pol['clean_text'] = pol.policy_text.map(clean)
pol.to_csv(A / 'policy_clauses_processed.csv', index=False)

print('[2/4] Word2Vec + TF-IDF')
corpus = tr.c1.tolist() + tr.c2.tolist() + pol.clean_text.tolist()
w2v = Word2Vec([s.split() for s in corpus if s.strip()], vector_size=100, window=5,
               min_count=2, workers=4, seed=42, epochs=10)
w2v.save(str(A / 'word2vec.model'))
pair_tfidf = TfidfVectorizer().fit(tr.c1.tolist() + tr.c2.tolist())
clause_tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit(pol.clean_text)
joblib.dump({'pair_tfidf': pair_tfidf, 'clause_tfidf': clause_tfidf}, A / 'vectorizers.joblib')

print('[3/4] NLI classifier (entailment / neutral / contradiction)')
def feats(df): return pair_features(df.c1.tolist(), df.c2.tolist(), w2v, pair_tfidf)
Xtr, Xdv, Xte = feats(tr), feats(dv), feats(te)
ytr, ydv, yte = [tr.gold_label.map(LABELS.index).values, dv.gold_label.map(LABELS.index).values,
                 te.gold_label.map(LABELS.index).values]
nli = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
                                     l2_regularization=1.0, early_stopping=True,
                                     validation_fraction=0.1, random_state=42).fit(Xtr, ytr)
pred = nli.predict(Xte)
metrics['nli'] = {
    'model': 'HistGradientBoosting on engineered features (overlap, TF-IDF cos, Word2Vec cos, negation, |a-b|, a*b)',
    'train_pairs': len(tr), 'dev_accuracy': round(accuracy_score(ydv, nli.predict(Xdv)), 4),
    'test_accuracy': round(accuracy_score(yte, pred), 4),
    'test_macro_f1': round(f1_score(yte, pred, average='macro'), 4),
    'labels': LABELS, 'confusion_matrix': confusion_matrix(yte, pred).tolist(),
    'report': classification_report(yte, pred, target_names=LABELS, output_dict=True)}
print('  NLI test acc', metrics['nli']['test_accuracy'])
joblib.dump(nli, A / 'nli_model.joblib')

print('[4/4] Fraud detector')
cl = pd.read_csv(D / 'insurance_fraud_claims.csv')
cl['collision_type'] = cl.collision_type.replace('?', 'Unknown')
cl['property_damage'] = cl.property_damage.replace('?', 'Unknown')
cl['police_report_available'] = cl.police_report_available.replace('?', 'Unknown')
CAT = ['incident_type', 'collision_type', 'incident_severity', 'authorities_contacted',
       'insured_hobbies', 'insured_occupation', 'property_damage', 'police_report_available',
       'insured_sex', 'insured_education_level', 'policy_state']
NUM = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
       'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim',
       'vehicle_claim', 'auto_year']
y = (cl.fraud_reported == 'Y').astype(int)
Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(cl[CAT + NUM], y, test_size=0.2, stratify=y, random_state=42)
pre = ColumnTransformer([('c', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CAT)],
                        remainder='passthrough')
fraud = Pipeline([('pre', pre), ('clf', HistGradientBoostingClassifier(
    max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=2.0,
    categorical_features=list(range(len(CAT))), random_state=42, class_weight='balanced'))])
fraud.fit(Xtr_f, ytr_f)
p = fraud.predict_proba(Xte_f)[:, 1]
metrics['fraud'] = {'model': 'HistGradientBoosting (balanced)', 'rows': len(cl),
    'fraud_rate': round(float(y.mean()), 3), 'test_auc': round(roc_auc_score(yte_f, p), 4),
    'test_accuracy': round(accuracy_score(yte_f, p > .5), 4), 'test_f1': round(f1_score(yte_f, p > .5), 4),
    'confusion_matrix': confusion_matrix(yte_f, p > .5).tolist()}
print('  fraud AUC', metrics['fraud']['test_auc'])
options = {c: sorted(set(map(str, cl[c].dropna().tolist()))) for c in CAT}
defaults = {c: (cl[c].median().item()) for c in NUM}
ranges = {c: [cl[c].min().item(), cl[c].max().item()] for c in NUM}
joblib.dump({'pipe': fraud, 'CAT': CAT, 'NUM': NUM, 'options': options, 'defaults': defaults,
             'ranges': ranges, 'modes': {c: cl[c].mode()[0] for c in CAT}}, A / 'fraud_model.joblib')
w = Word2Vec.load(str(A / 'word2vec.model'))
metrics['w2v'] = {'vocab': len(w.wv.index_to_key),
                  'similar_cover': w.wv.most_similar('cover', topn=5) if 'cover' in w.wv else [],
                  'similar_damage': w.wv.most_similar('damage', topn=5) if 'damage' in w.wv else []}
json.dump(metrics, open(A / 'metrics.json', 'w'), indent=2, default=float)
print('Done. Artifacts in', A)
