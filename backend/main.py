"""InsureClaim AI - FastAPI backend.  Run:  uvicorn main:app --app-dir backend --port 8000"""
import json, pickle
from pathlib import Path
import joblib, numpy as np, pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gensim.models import Word2Vec
from nlp import clean, preprocess, pair_features, sent_vec

ROOT = Path(__file__).resolve().parent.parent
A, D, FE = ROOT / 'artifacts', ROOT / 'data', ROOT / 'frontend'
LABELS = ['entailment', 'neutral', 'contradiction']

# ---------- load artifacts ----------
w2v = Word2Vec.load(str(A / 'word2vec.model'))
vecs = joblib.load(A / 'vectorizers.joblib')
pair_tfidf, clause_tfidf = vecs['pair_tfidf'], vecs['clause_tfidf']
nli_model = joblib.load(A / 'nli_model.joblib')
FR = joblib.load(A / 'fraud_model.joblib')
fraud_pipe, CAT, NUM = FR['pipe'], FR['CAT'], FR['NUM']
metrics = json.load(open(A / 'metrics.json'))
clauses = pd.read_csv(A / 'policy_clauses_processed.csv')
clause_mat = clause_tfidf.transform(clauses.clean_text)
clause_vecs = np.vstack([sent_vec(t, w2v) for t in clauses.clean_text])
claims_raw = pd.read_csv(D / 'insurance_fraud_claims.csv')

# optional BiLSTM from the notebook (Assignment 4) - used automatically if trained
bilstm = None
if (A / 'bilstm.keras').exists() and (A / 'tokenizer.pkl').exists():
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        bilstm = tf.keras.models.load_model(A / 'bilstm.keras')
        bilstm_tok = pickle.load(open(A / 'tokenizer.pkl', 'rb'))
        print('BiLSTM loaded - ensemble mode')
    except Exception as e:  # noqa
        print('BiLSTM not loaded:', e)

app = FastAPI(title='InsureClaim AI', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


# ---------- NLI ----------
def nli_probs(premises, hyps):
    c1, c2 = [clean(p) for p in premises], [clean(h) for h in hyps]
    p = nli_model.predict_proba(pair_features(c1, c2, w2v, pair_tfidf))
    if bilstm is not None:
        X1 = pad_sequences(bilstm_tok.texts_to_sequences(c1), maxlen=25, padding='post')
        X2 = pad_sequences(bilstm_tok.texts_to_sequences(c2), maxlen=25, padding='post')
        p = (p + bilstm.predict([X1, X2], verbose=0)) / 2
    return p


def pdict(row): return {l: round(float(v), 4) for l, v in zip(LABELS, row)}


def retrieve(query, k=5):
    q = clean(query)
    s_tf = (clause_tfidf.transform([q]) @ clause_mat.T).toarray().ravel()
    qv = sent_vec(q, w2v)
    s_w = np.array([float(qv @ v / ((np.linalg.norm(qv) * np.linalg.norm(v)) or 1)) for v in clause_vecs])
    score = 0.7 * s_tf + 0.3 * np.clip(s_w, 0, 1)
    return score, np.argsort(-score)[:k]


def clause_json(i, rel):
    r = clauses.iloc[i]
    return {'id': r.clause_id, 'category': r.category, 'type': r.clause_type,
            'text': r.policy_text, 'relevance': round(float(rel), 4)}


# ---------- fraud ----------
def build_row(f):
    row = {}
    for c in CAT:
        v = str(f.get(c, FR['modes'][c]))
        row[c] = v if v in FR['options'][c] else FR['modes'][c]
    for c in NUM:
        try: row[c] = float(f.get(c, FR['defaults'][c]))
        except (TypeError, ValueError): row[c] = FR['defaults'][c]
    return row


def fraud_score(row):
    return float(fraud_pipe.predict_proba(pd.DataFrame([row])[CAT + NUM])[0, 1])


def explain_fraud(row, p):
    out = []
    for c in CAT + NUM:
        alt = dict(row); alt[c] = FR['modes'][c] if c in CAT else FR['defaults'][c]
        d = p - fraud_score(alt)
        if abs(d) > 0.01:
            out.append({'feature': c, 'value': row[c], 'impact': round(d, 4)})
    return sorted(out, key=lambda x: -abs(x['impact']))[:6]


def narrative(r):
    sev = r['incident_severity'].lower()
    t = f"The insured vehicle was involved in a {r['incident_type'].lower()}"
    if r['collision_type'] not in ('Unknown', 'nan') and 'Theft' not in r['incident_type'] and 'Parked' not in r['incident_type']:
        t += f" ({r['collision_type'].lower()})"
    t += f" causing {sev}. "
    t += (f"{r['authorities_contacted'].replace('Other', 'Other authorities')} were contacted. " if r['authorities_contacted'] != 'None'
          else "No authorities were contacted. ")
    t += {'YES': 'A police report is available. ', 'NO': 'No police report is available. '}.get(
        r['police_report_available'], 'Police report status is unknown. ')
    if int(r['bodily_injuries']): t += f"{int(r['bodily_injuries'])} bodily injuries were reported. "
    if r['property_damage'] == 'YES': t += 'Property damage was reported. '
    t += f"There were {int(r['witnesses'])} witnesses and {int(r['number_of_vehicles_involved'])} vehicles involved. "
    t += f"The insured enjoys {r['insured_hobbies']} as a hobby. Total claim amount is {int(r['total_claim_amount'])}."
    return t


HIGH_RISK_HOBBIES = {'skydiving', 'paintball', 'base-jumping', 'bungee-jumping', 'racing'}
TYPE_CAT = {'Single Vehicle Collision': 'PC002', 'Multi-vehicle Collision': 'PC003',
            'Vehicle Theft': 'PC006', 'Parked Car': 'PC030'}


def rules(r, p_fraud):
    """Deterministic policy rules -> {clause_id: (status, reason)}; status in ok/condition/risk."""
    R = {}
    def add(cid, status, why):
        order = {'ok': 0, 'condition': 1, 'risk': 2}
        if cid not in R or order[status] > order[R[cid][0]]: R[cid] = (status, why)
    it, sev = r['incident_type'], r['incident_severity']
    if it in TYPE_CAT: add(TYPE_CAT[it], 'ok', f'Incident type "{it}" matches this coverage clause.')
    if it in ('Single Vehicle Collision', 'Multi-vehicle Collision'):
        add('PC001', 'ok', 'Collision damage covered regardless of fault, subject to deductible.')
        add('PC023', 'condition', f"Deductible of {int(r['policy_deductable'])} applies before payout.")
        if r['collision_type'] == 'Unknown':
            add('PC005', 'risk', 'Point of impact is missing from the claim - required for processing.')
    if it == 'Vehicle Theft':
        if r['police_report_available'] != 'YES':
            add('PC008', 'risk', 'Theft claim without a valid police report will be denied.')
            add('PC006', 'risk', 'Coverage requires a police report within 24h.')
        else: add('PC008', 'ok', 'Police report present.')
    if r['insured_hobbies'] in HIGH_RISK_HOBBIES:
        add('PC018', 'risk', f"Hobby \"{r['insured_hobbies']}\" may fall under high-risk activity exclusions - verify context.")
    if int(r['bodily_injuries']) > 0:
        add('PC012', 'ok', 'Bodily injury reported in a collision claim.')
        add('PC014', 'condition', 'Medical documentation required for injury claims above minor threshold.')
        if r['authorities_contacted'] == 'None':
            add('PC020', 'risk', 'Injury reported but no authorities contacted.')
    if r['property_damage'] == 'YES':
        add('PC009', 'ok', 'Third-party property damage reported.')
        add('PC011', 'condition', 'Photographic/documentary evidence needed.')
    if sev == 'Total Loss':
        add('PC015', 'ok', 'Total loss settlement at pre-incident market value.')
        add('PC016', 'condition', 'Independent damage assessment required.')
    if sev in ('Minor Damage', 'Trivial Damage'): add('PC017', 'ok', 'Minor damage - reduced payout applies.')
    if int(r['witnesses']) > 0: add('PC022', 'condition', 'Witness contact details must be provided.')
    if float(r['umbrella_limit']) > 0 and sev in ('Major Damage', 'Total Loss'):
        add('PC027', 'ok', 'Umbrella liability coverage may extend to this major incident.')
    if p_fraud >= 0.5:
        add('PC029', 'risk', f'Fraud model score {p_fraud:.0%}: insurer may investigate before payout.')
    if p_fraud >= 0.75:
        add('PC028', 'risk', 'Very high fraud score: possible misrepresentation / staged incident.')
    return R


# ---------- API ----------
@app.get('/api/health')
def health(): return {'status': 'ok', 'engine': 'gbm+bilstm' if bilstm else 'gbm', 'clauses': len(clauses)}


@app.get('/api/metrics')
def get_metrics(): return metrics


@app.get('/api/clauses')
def get_clauses(): return [clause_json(i, 0) for i in range(len(clauses))]


@app.get('/api/schema')
def schema(): return {'categorical': FR['options'], 'numeric_defaults': FR['defaults'], 'numeric_ranges': FR['ranges']}


@app.get('/api/samples')
def samples():
    df = pd.concat([claims_raw[claims_raw.fraud_reported == 'Y'].sample(3, random_state=7),
                    claims_raw[claims_raw.fraud_reported == 'N'].sample(3, random_state=7)])
    cols = CAT + NUM
    out = []
    for _, r in df.iterrows():
        d = {c: (r[c].item() if hasattr(r[c], 'item') else r[c]) for c in cols}
        for c in ('collision_type', 'property_damage', 'police_report_available'):
            if d[c] == '?': d[c] = 'Unknown'
        out.append({'label': f"{d['incident_type']} · {d['incident_severity']} · {int(d['total_claim_amount']):,}",
                    'actual_fraud': r['fraud_reported'], 'fields': d})
    return out


@app.post('/api/preprocess')
def api_preprocess(body: dict = Body(...)):
    s, t, c = preprocess(body.get('text', ''))
    return {'sentences': s, 'tokens': t, 'clean_text': c}


@app.post('/api/nli')
def api_nli(body: dict = Body(...)):
    p, h = str(body.get('premise', '')).strip(), str(body.get('hypothesis', '')).strip()
    if not p or not h: raise HTTPException(400, 'premise and hypothesis are required')
    pr = nli_probs([p], [h])[0]
    return {'probs': pdict(pr), 'label': LABELS[int(np.argmax(pr))],
            'premise_clean': clean(p), 'hypothesis_clean': clean(h)}


@app.post('/api/search')
def api_search(body: dict = Body(...)):
    q = str(body.get('query', '')).strip()
    if not q: raise HTTPException(400, 'query required')
    score, idx = retrieve(q, int(body.get('k', 5)))
    pr = nli_probs([clauses.policy_text[i] for i in idx], [q] * len(idx))
    return [{**clause_json(i, score[i]), 'nli': pdict(pr[j])} for j, i in enumerate(idx)]


@app.post('/api/claim')
def api_claim(body: dict = Body(...)):
    row = build_row(body)
    p = fraud_score(row)
    text = str(body.get('narrative') or '').strip() or narrative(row)
    R = rules(row, p)
    score, idx = retrieve(text, 6)
    ids = list(dict.fromkeys([clauses.index[clauses.clause_id == c][0] for c in R] + [int(i) for i in idx]))[:10]
    probs = nli_probs([clauses.policy_text[i] for i in ids], [text] * len(ids))
    out = []
    for j, i in enumerate(ids):
        cid = clauses.clause_id[i]
        item = {**clause_json(i, score[i]), 'nli': pdict(probs[j]), 'rule': None}
        if cid in R: item['rule'] = {'status': R[cid][0], 'reason': R[cid][1]}
        out.append(item)
    out.sort(key=lambda x: ({'risk': 0, 'condition': 1, 'ok': 2, None: 3}[x['rule']['status'] if x['rule'] else None], -x['relevance']))
    risks = [c for c in out if c['rule'] and c['rule']['status'] == 'risk']
    excl = [c for c in risks if c['type'] == 'EXCLUSION']
    if p >= 0.75: verdict = ('ESCALATE TO SIU', 'danger', 'Very high fraud probability - route to the special investigations unit.')
    elif excl: verdict = ('LIKELY EXCLUDED', 'danger', 'An exclusion clause appears to apply - manual review required.')
    elif risks or p >= 0.5: verdict = ('MANUAL REVIEW', 'warn', 'Condition violations or elevated fraud risk detected.')
    else: verdict = ('LIKELY COVERED', 'ok', 'No exclusions triggered; standard conditions apply.')
    lvl = 'High' if p >= 0.6 else 'Medium' if p >= 0.35 else 'Low'
    s, t, c = preprocess(text)
    return {'fraud': {'probability': round(p, 4), 'level': lvl, 'factors': explain_fraud(row, p)},
            'narrative': text, 'preprocessing': {'tokens': t, 'clean_text': c, 'sentences': len(s)},
            'verdict': {'label': verdict[0], 'tone': verdict[1], 'summary': verdict[2]}, 'clauses': out}


app.mount('/static', StaticFiles(directory=FE), name='static')


@app.get('/')
def index(): return FileResponse(FE / 'index.html')
