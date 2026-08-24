# University Inquiry Chatbot

BMCS Artificial Intelligence assignment, Topic 5: Chatbot Development.

The prototype compares two independently implemented approaches to the same university FAQ scenario:

1. **Member 1 - Google Dialogflow ES platform:** an importable agent configuration with intents, entities, eight parameters, required-slot prompting, static responses, and fallback handling.
2. **Member 2 - local machine learning:** a Python pipeline using character-boundary TF-IDF features (3-5 character n-grams) and balanced Logistic Regression with a confidence threshold and a reviewed-query log.

> Replace the member placeholders in the final report with the real names and student IDs before submission.

## Important evaluation note

The reproducible offline benchmark uses a **training-only intent-matching simulator** as a local Dialogflow-style baseline. It is not described as a Google cloud result. A genuine Dialogflow ES score requires importing the supplied agent, training it without the held-out test utterances, and calling Dialogflow `detectIntent` on those unseen utterances. Do not combine training and test phrases in the same agent before measuring accuracy.

The evaluation code reports intent Accuracy, weighted Precision, Recall, F1, coverage, and fallback rate. BLEU and ROUGE-1 F1 are calculated separately from `data/response_quality_test.json`, whose queries and source-grounded reference answers do not occur in the training data. Reusing the same intent response as both prediction and reference is explicitly rejected by validation tests.

Formal user-satisfaction results are read only from the frozen, anonymized Google Forms snapshot in `data/user_feedback_verified.json` (N=5, collected 12-24 August 2026). Local UI demo submissions are written to ignored `data/user_feedback.json` and are excluded from the report. A favorable rating is defined in advance as 4 or 5; it is not calculated by dividing the mean by five.

## Setup

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If WordNet data is not already available, the preprocessing module safely falls back to the original token. To enable lemmatisation explicitly:

```bash
python -m nltk.downloader wordnet omw-1.4
```

## Run the prototype

Set an administrator PIN before enabling active-learning mutations:

```bash
export CHATBOT_ADMIN_PIN='choose-a-demo-pin'
streamlit run app.py
```

Without `CHATBOT_ADMIN_PIN`, the Active Learning Review tab remains read-only. This prevents accidental dataset deletion or modification during a public demo.

## Reproduce the evidence

```bash
python evaluate.py
python auto_tester.py
python -m unittest discover -s tests -v
python src/create_dialogflow_zip.py
unzip -t dialogflow_agent.zip
```

Generated evaluation data is written under `data/`. The automated probe contains expected intent labels and records pass/fail status instead of treating arbitrary responses as successful tests.

## Dialogflow ES

1. Create a Dialogflow ES agent using English and the `Asia/Kuala_Lumpur` time zone.
2. Import `dialogflow_agent.zip` from **Settings -> Export and Import -> Import from ZIP**.
3. Review the custom entities, parameters, required prompt, and welcome/fallback intents.
4. Confirm that webhook fulfillment remains disabled; this submission uses controlled static responses.
5. Capture screenshots of the imported intents/entities and a real test-console conversation for the presentation/report evidence appendix.

The export package proves that the configuration is reproducible; it does not by itself prove that a cloud agent was deployed or tested.

## Main files

- `app.py` - Streamlit interface, benchmark dashboard, feedback form, and protected review UI.
- `src/ml_model.py` - local character TF-IDF + Logistic Regression classifier.
- `src/dialogflow_client.py` - explicitly labelled offline/train-only matching baseline.
- `evaluate.py` - leakage-free stratified evaluation and exported metrics.
- `auto_tester.py` - labelled regression/health probes.
- `data/intents.json` - curated intent patterns and controlled responses.
- `data/response_quality_test.json` - independent, source-grounded response references.
- `data/user_feedback_verified.json` - anonymized, verified five-item Google Forms snapshot used by the dashboard and report.
- `src/create_dialogflow_zip.py` - deterministic Dialogflow ES export builder.
- `dialogflow_agent.zip` - importable Dialogflow ES agent package.
- `docs/rubric_compliance.md` - requirement-to-evidence audit and final human-action checklist.
- `AI Report - Final.docx` - final documentation after it is generated and checked.
- `Google Docs Copy - Report Content.md` - copy-friendly text and tables exported from the same final DOCX.

## Submission checklist

- Replace all `[TO BE PROVIDED]` identity fields in the report.
- Complete and sign one plagiarism statement per student.
- Add authentic Dialogflow Console screenshots after import.
- Verify that the Google Forms response count still matches the frozen N=5 snapshot, or deliberately refresh the anonymized snapshot and rebuild the report.
- Upload `AI Report - Final.docx` to Google Drive and open it with Google Docs, or copy from `Google Docs Copy - Report Content.md`; then insert the supplied figures and complete all identity/signature placeholders.
- Demonstrate each member's own implementation and be ready for on-the-spot code changes.
- Submit source files and `dialogflow_agent.zip`; exclude `myenv/`, `.venv/`, caches, temporary renders, and secrets.
