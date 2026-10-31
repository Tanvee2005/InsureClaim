# 🛡️ InsureClaim AI

End-to-end NLP project (from `NLP_Project.ipynb`): **insurance claim ↔ policy clause matching with Natural Language Inference + fraud detection**, served by a FastAPI backend with a neon single-page frontend.

## Quick start
```bash
pip install -r requirements.txt
python run.py            # opens http://localhost:8000  (pre-trained artifacts are included)
```
Or: `uvicorn main:app --app-dir backend --port 8000` · Docker: `docker build -t insureclaim . && docker run -p 8000:8000 insureclaim`

## What it does
| Tab | Feature |
|---|---|
| Claim Analyzer | Structured claim → fraud probability (+ per-feature explanation), auto-generated narrative, deterministic policy rules, NLI (entailment/neutral/contradiction) against relevant clauses, final verdict (Likely Covered / Manual Review / Likely Excluded / Escalate to SIU) |
| NLI Playground | Any premise/hypothesis pair → 3-class probabilities |
| Clause Search | Free-text situation → TF-IDF + Word2Vec retrieval over the 30 curated clauses, each scored with NLI |
| Policy Library | Browse/filter all clauses |
| Model Insights | Accuracy, F1, AUC, confusion matrices, pipeline |

## Pipeline (mirrors the notebook)
1. **Preprocessing** – sentence/word tokenise, lowercase, strip URLs/punct, stop-words removed **but negations kept**, POS-aware WordNet lemmatisation (`backend/nlp.py`).
2. **Features** – TF-IDF (1–2 grams) for clauses, Word2Vec (100-d, trained on SNLI + clauses), pair features: word overlap, hypothesis coverage, length diff, negation mismatch, TF-IDF cosine, W2V cosine, |a−b|, a⊙b.
3. **NLI model** – HistGradientBoosting on pair features (fast, CPU-only). **Optional BiLSTM** from Assignment 4: `pip install tensorflow && python backend/train_bilstm.py` → API auto-ensembles it.
4. **Fraud model** – HistGradientBoosting on the 1,000 claims (balanced), explanations via occlusion (replace feature with median/mode and measure score change).
5. **Rule engine** – clause-specific checks (police report for theft, high-risk hobbies, injury without authorities, deductible, witnesses…).

## API
`GET /api/health · /api/schema · /api/samples · /api/clauses · /api/metrics` · `POST /api/claim · /api/nli · /api/search · /api/preprocess` — interactive docs at `/docs`.

## Retraining
`python backend/train.py` (uses `data/snli/*.csv` samples: 40k train / 4k dev / 4k test). For the full SNLI run, replace the CSVs (columns `sentence1,sentence2,gold_label`) – accuracy rises with more data.

## Honest notes
* Bundled NLI is a lightweight feature model (~61% on a 4k SNLI test sample, 3-class). The BiLSTM and more data improve this; the rule engine covers the clear-cut policy logic.
* Fraud AUC ≈ 0.83 on a held-out 20% split of a small (1,000-row) public dataset – a decision-support tool, **not** an automated claim denial system.
* Test accuracy is on SNLI-style text, not on real policy language; clause matching was validated on the curated clause set only.
* NLTK data downloads automatically on first run (needs internet once).
