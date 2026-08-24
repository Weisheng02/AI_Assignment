# Requirement and Rubric Compliance Check

This checklist maps the frozen submission snapshot to the assignment specification. It distinguishes implemented evidence from items that require genuine student or external evidence.

## Topic 5: Chatbot Development requirements

| Requirement | Evidence | Status |
|---|---|---|
| University FAQ scenario and background study | Final report Sections 1–2; official source register in `data/sources.md` | Met |
| Two different member implementations | Dialogflow ES export (`dialogflow_agent.zip`) and local character TF-IDF + Logistic Regression pipeline (`src/ml_model.py`) | Met technically; add real member identities |
| FAQ and natural-language interaction | Streamlit chat UI and 36-intent dataset | Met |
| Dataset collection, preprocessing and representation | `data/intents.json`, `src/preprocessing.py`, character-boundary TF-IDF | Met |
| ML intent model and controlled responses | `src/ml_model.py`; confidence gate and fallback at 0.20 | Met |
| Platform intents, entities and responses | 36 source intents, 4 custom entities, 8 parameters and fallback/welcome intents in the validated Dialogflow ZIP | Met as an importable artifact; authenticated cloud testing remains external |
| Intent-recognition testing | Leakage-free stratified 80/20 split: 342 train / 86 test, zero cleaned-text overlap | Met |
| Accuracy, precision, recall and weighted F1 | Saved in `data/evaluation_results.json` and reported in the performance dashboard/report | Met |
| BLEU or ROUGE response-quality evaluation | 14 independent source-grounded cases; training-query and training-response reuse are rejected by tests | Met |
| User feedback and satisfaction | Survey instrument and empty verified feedback store | Instrument met; genuine respondents and statistics still required |

## Documentation rubric (40%)

| Rubric item | Final-report coverage | Status |
|---|---|---|
| Introduction | Background, problem statement, research gap, aligned objectives and significance | Met |
| Related Work | Critical comparison of platform, retrieval and ML approaches with cited limitations | Met |
| Methodology | System flow, dataset provenance, preprocessing, both implementations, model selection and metric definitions | Met |
| Results & Discussion | Held-out metrics, coverage/fallback trade-offs, independent response scores, error analysis and limitations | Met |
| Conclusion, References & Sources | Achievements, limitations, future work, APA-style references, official source register and appendices | Met |

## Prototype rubric (60%)

| Rubric item | Evidence | Status |
|---|---|---|
| User interface / output | Five-tab Streamlit interface, model selector, chat evidence, metrics dashboard and protected review workflow | Met |
| Programming | Modular source, atomic JSON writes, validation, confidence handling and 22 automated tests | Met |
| Degree of completion | Local solution runs end to end; 10/10 labelled probes pass for both local clients; Streamlit AppTest has zero exceptions | Met locally |
| System implementation | Prototype and report describe the same two-approach architecture and frozen metrics | Met |
| Presentation and on-the-spot coding | Demonstration checklist and contribution appendix are supplied | Assessed live; both students must prepare |

## Frozen evidence snapshot

- Evaluation set: 428 unique, unambiguous phrases across 36 intents.
- Member 1 offline Dialogflow-style baseline: accuracy 0.6163; weighted F1 0.5983; coverage 0.9186.
- Member 2 character TF-IDF + Logistic Regression: accuracy 0.7442; weighted F1 0.7626; coverage 0.8721.
- Independent response set: 14 cases; Member 1 BLEU 0.1460 / ROUGE-1 F1 0.4086; Member 2 BLEU 0.1689 / ROUGE-1 F1 0.4137.
- Automated checks: 22/22 tests, 10/10 labelled probes for each local client, valid Dialogflow ZIP, and zero Streamlit AppTest exceptions.

## Required human actions before submission

1. Replace every `[TO BE PROVIDED]` identity field with the real names, student IDs, tutorial group and tutor.
2. Each student must review, complete and sign their own plagiarism statement.
3. Import the agent ZIP into the team's Dialogflow ES account and capture authenticated console/API evidence. Do not relabel the offline simulator metric as Google cloud accuracy.
4. Collect voluntary, genuine survey responses before reporting a sample size or satisfaction statistic. Keep `data/user_feedback.json` empty until verified responses exist.
5. Both members must rehearse their own implementation and be ready for the live demonstration, Q&A and on-the-spot coding component.
