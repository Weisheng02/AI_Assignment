<!-- Copy this file into Google Docs, or upload AI Report - Final.docx directly to Google Drive. -->

BMCS2003  •  ARTIFICIAL INTELLIGENCE  •  SESSION 202605

TAR UMT University
Inquiry Chatbot

Dual-Approach Design and Leakage-Free Evaluation of a Dialogflow ES Configuration and a Local TF-IDF + Logistic Regression Pipeline

PREPARED BY  Member 1: [TO BE PROVIDED]  |  Student ID: [TO BE PROVIDED]

Member 2: [TO BE PROVIDED]  |  Student ID: [TO BE PROVIDED]

TUTORIAL GROUP  [TO BE PROVIDED]

TUTOR  [TO BE PROVIDED]

SUBMISSION DEADLINE  28 August 2026, before 12:00 p.m.

EVALUATION SNAPSHOT  2026-08-24T18:02:56.633862

# Document control and report map

Evidence boundary  All reported model values are read from data/evaluation_results.json. Member 1 values describe a local, train-only Dialogflow-style simulator and are not cloud Dialogflow metrics. Survey results are read from the anonymized Google Forms snapshot in data/user_feedback_verified.json; favorable means a rating of 4 or 5. BLEU and ROUGE are reported only when the evaluation artifact contains scores derived from independent reference answers.

*Table 1: Document control*

| Field | Recorded value |
| --- | --- |
| Report builder | tools/build_final_report.py |
| Output | AI Report - Final.docx |
| Evaluation source | data/evaluation_results.json |
| Evaluation generated at | 2026-08-24T18:02:56.633862 |
| Document built at | 2026-08-24T20:25:35+08:00 |
| Identifying details | Member names, IDs, tutorial group, and tutor remain [TO BE PROVIDED]. |

## Report map

- Executive summary and rubric traceability

- 1. Introduction

- 2. Related Work

- 3. Methodology

- 4. Results and Discussion

- 5. Conclusion, References and Sources

- Appendices A–F: contributions, reproducibility, survey, authentic evidence, and two plagiarism forms

The five numbered sections mirror the five Documentation Assessment Rubric items in Appendix 1 of the assignment specification.

# Executive summary

This project addresses repetitive and fragmented university enquiries by designing a task-oriented chatbot for TAR UMT information. Two development approaches are represented: a Dialogflow ES configuration path and a local Python path. The local path cleans and lemmatises queries with NLTK, extracts character-boundary TF-IDF features (3–5 character n-grams), and applies balanced multinomial Logistic Regression (C = 30, max_iter = 2000) with a 0.20 confidence threshold. Low-confidence queries return a controlled fallback and can be logged for human review.

The current evidence snapshot uses a stratified 80/20 split with random_state = 42, 342 training examples, 86 held-out examples, and 0 cleaned-text overlaps between partitions. The highest held-out accuracy in the recorded comparison is 76.74% for Baseline 2 (Linear SVM). Member 2 records 74.42% accuracy, 0.7626 weighted F1, 87.21% coverage, and 12.79% fallback rate. These values demonstrate the present prototype's limitations rather than a production-ready result.

Response-generation scoring status is Independent reference test (14 cases); intent accuracy=0.7857; coverage=0.9286. The verified usability survey contains 5 anonymous response(s) and 25 item ratings. The five-item mean is 3.76/5 and 17/25 ratings (68.0%) are favorable (4 or 5). These are preliminary descriptive findings because N=5 is too small for generalisation.

## Rubric traceability

*Table 2: Documentation-rubric traceability*

| Rubric item | Primary evidence in this report | Good-band intent |
| --- | --- | --- |
| Introduction | Background, problem, gap, aligned objectives, scope, significance | Comprehensive and justified |
| Related Work | Critical comparison of university chatbots, Dialogflow, local ML, and evaluation | Evaluated, not merely described |
| Methodology | Architecture, dataset provenance, algorithms, split, metrics, and validity controls | Logical, reproducible, justified |
| Results & Discussion | Dynamic tables and charts, confusion analysis, implications, limitations | Evidence-led interpretation |
| Conclusion & References & Source | Achievements, limitations, future work, APA references, source inventory | Complete and academically transparent |

# 1. Introduction

## 1.1 Background and context

University applicants and students routinely ask recurring questions about programmes, admissions, fees, academic calendars, accommodation, facilities, student affairs, and contact channels. The answers may exist on official pages, but users must still identify the correct page, interpret institutional terminology, and verify that the information is current. A task-oriented FAQ chatbot can shorten this search path by mapping a natural-language query to a controlled intent and then returning a curated answer or an official source link.

Campus chatbots are a credible application area rather than a purely technical exercise. Ranoliya et al. (2017) demonstrated a university FAQ chatbot, while Dibitonto et al. (2018) situated a virtual assistant within student university life. Dialogflow ES provides managed concepts for agents, intents, entities, contexts, fulfilment, and integrations (Google Cloud, n.d.). A local statistical classifier offers a contrasting path whose data flow, features, decision threshold, and errors can be inspected in the repository.

## 1.2 Problem statement

The practical problem is not simply to return an answer when a familiar phrase is entered. The system must recognise paraphrases across many closely related university intents, reject low-confidence queries safely, preserve the provenance of factual answers, and provide evidence that its reported metrics were obtained without training–test leakage. Closely related labels—such as admission versus admission documents, location versus campus map, and facilities versus sports—make the task difficult when each intent has only a small number of examples.

A second problem is evidence quality. A configured Dialogflow agent, a local approximation of Dialogflow-style pattern matching, and a deployed local classifier are different systems. Reporting an offline simulator score as cloud accuracy would invalidate the comparison. Likewise, BLEU, ROUGE, and satisfaction statistics require independent references or genuine respondents; template self-comparison and invented responses are not acceptable evidence.

## 1.3 Research gap

The reviewed studies establish the usefulness of university chatbots and the importance of selecting an appropriate development platform, but they do not provide a leakage-free, same-split comparison for this repository's TAR UMT inquiry taxonomy. This project therefore focuses on an auditable comparison in which the evaluation boundary is explicit: a train-only local Dialogflow-style simulator, the deployed-configuration local Logistic Regression model, and two classical baselines are tested on the same held-out queries. Cloud Dialogflow performance remains a separate validation task requiring authentic console or API evidence.

## 1.4 Objectives

Design a university FAQ chatbot that covers major TAR UMT inquiry categories and returns controlled, source-aware responses.

Represent two distinct member approaches: a Dialogflow ES agent configuration and an offline Python intent classifier.

Implement the local classifier with deterministic preprocessing, character-boundary TF-IDF features, balanced Logistic Regression, confidence gating, and fallback logging.

Evaluate intent recognition on a fixed stratified held-out split using accuracy, weighted precision, weighted recall, weighted F1, coverage, fallback rate, and confusion analysis.

Evaluate response quality using independent references and analyse genuine user-satisfaction responses with transparent descriptive rules.

Deliver reproducible source code, evaluation artifacts, charts, and documentation that can be regenerated from the repository.

## 1.5 Scope and significance

The current source inventory contains 36 semantic intents and 432 raw training phrases in data/intents.json. The chatbot is English-first, single-turn, and task-oriented. Responses are selected from controlled templates rather than generated freely. This scope supports predictable answers and reduces hallucination risk, but it does not replace official TAR UMT pages or staff advice. Time-sensitive fees, dates, policies, and contact details must be verified at the linked official source.

The project is significant in three ways. For users, it offers a consistent entry point to common information. For administrators, fallback logs expose unanswered demand that can guide dataset maintenance. For AI study, the project demonstrates how thresholding, data provenance, and evaluation design can matter as much as the choice of classifier.

Success criterion  A working prototype is necessary but not sufficient. Success means that system claims remain traceable to code, data, and authentic test evidence, and that limitations are disclosed where the current scores are weak.

# 2. Related Work

## 2.1 University FAQ and campus assistants

Ranoliya et al. (2017) presented a chatbot for university-related FAQs, illustrating the fit between structured institutional questions and intent/pattern-oriented interaction. Its relevance to this project is the bounded FAQ domain; its limitation for the present study is that a system that performs well on known patterns may still generalise poorly to held-out paraphrases. Dibitonto et al. (2018) designed LiSA as a campus virtual assistant to support students in university life. That work broadens the design question from classification alone to the student's situated experience and reinforces the need for usable, context-appropriate responses.

## 2.2 Platform-based and local development

Dialogflow ES is a managed natural-language-understanding platform for conversational interfaces. Its agent model organises intents, entities, responses, fulfilment, contexts, and integrations (Google Cloud, n.d.). This can accelerate configuration and integration, but the trained cloud service is externally managed and must be evaluated through its real console or API. A local simulator can exercise repository patterns and provide a transparent baseline, but it cannot stand in for Google's NLU.

Pérez-Soler et al. (2021) frame chatbot development as a tool-selection problem. Their comparison-oriented perspective is useful because the right choice depends on requirements such as deployment control, integration, language support, cost, and maintainability. The local TF-IDF pipeline used here is deliberately inspectable and offline-capable. Character-boundary features improve tolerance to small spelling and word-form variations, but they still require representative data and do not supply semantic understanding.

## 2.3 Evaluation of classification and responses

Accuracy alone can obscure minority-class behaviour in a multi-intent dataset. Weighted precision, recall, and F1 summarise class-level outcomes while preserving class prevalence; the confusion matrix shows which labels are exchanged. Coverage and fallback rate are also essential for a thresholded chatbot because an apparently cautious model can improve the quality of answered cases by declining many inputs.

BLEU measures n-gram precision with a brevity penalty (Papineni et al., 2002), while ROUGE includes recall-oriented overlap measures (Lin, 2004). Both require candidate outputs and independent reference answers. For a retrieval-style chatbot with template responses, comparing a selected template against the same template bank would be circular. The present report therefore displays N/A when independent reference scoring is absent and treats human judgement as a complementary future measure.

*Table 3: Critical comparison of prior work and the present study*

| Source | Contribution | Relevant limitation or trade-off | Design implication here |
| --- | --- | --- | --- |
| Ranoliya et al. (2017) | University FAQ chatbot | Pattern success need not imply paraphrase generalisation | Use a held-out split and confusion analysis |
| Dibitonto et al. (2018) | Campus assistant designed around student life | User experience extends beyond classifier accuracy | Include authentic usability protocol |
| Pérez-Soler et al. (2021) | Framework for choosing chatbot tools | No single tool dominates every deployment criterion | Separate cloud configuration from local ML evidence |
| Google Cloud (n.d.) | Dialogflow ES agent, intent, entity, fulfilment, and integration concepts | Cloud behaviour must be tested in the actual service | Do not label a local simulator score as cloud accuracy |
| Papineni et al. (2002); Lin (2004) | Automatic text-overlap metrics | Scores depend on independent references and do not establish factuality | Report N/A until a valid reference test is executed |

## 2.4 Synthesis and justification

The literature supports a dual-approach project but also exposes the main risk: unlike systems can be compared as if their evidence were equivalent. The present methodology addresses that risk by identifying each execution path, using identical held-out queries only for local evaluation components, recording coverage alongside accuracy, and retaining cloud validation as an explicit evidence gap. This makes the comparison more cautious but more defensible.

# 3. Methodology

## 3.1 Research design and requirements mapping

The project follows a design–build–evaluate workflow. Requirements were translated into a bounded FAQ scenario, two development paths, an intent dataset, controlled response templates, a user interface, and a reproducible evaluation harness. The evaluation is observational: it reports current prototype behaviour and does not claim statistical generalisation beyond the held-out sample.

*Table 4: Chatbot-assignment requirement mapping*

| Assignment requirement | Repository implementation | Evidence status |
| --- | --- | --- |
| Real-life chatbot scenario | TAR UMT FAQ and student-information assistant | Implemented |
| Background study | Section 2 compares university chatbots, tools, and metrics | Documented |
| Development approach | Dialogflow ES configuration artifacts plus local Python ML | Artifacts present; cloud run needs authentic evidence |
| Different member solutions | Member 1 configuration/simulator track; Member 2 TF-IDF + LR track | Contribution ownership must be confirmed |
| Intent-recognition testing | Leakage-free split, four-model metrics, confusion analysis | Measured |
| Response relevancy/quality | Independent-reference protocol | Measured from 14 cases |
| User satisfaction | Five-item Google Forms instrument; verified anonymous snapshot; Section 4.4 and Appendix C | Measured descriptively; N=5 |

## 3.2 System architecture

*Figure 1: Operational architecture and offline-evaluation boundary*

The architecture distinguishes operation from evaluation. On the Dialogflow path, the user query is sent to the configured Google service; it does not first pass through the local NLTK pipeline. On the local path, preprocessing, vectorisation, classification, thresholding, response retrieval, and fallback logging execute within Python. The evaluation harness creates the train/test split and constructs the local models from training examples only.

## 3.3 Dataset, provenance, and representation

*Table 5: Dataset inventory and evaluation snapshot*

| Artifact or quantity | Current value | Interpretation |
| --- | --- | --- |
| data/intents.json | 36 semantic intents; 432 raw phrases; 36 response templates | Current source inventory |
| Evaluation rows | 428 | Canonical rows recorded when evaluation_results.json was generated |
| Training partition | 342 | 80% stratified; models fit only here |
| Held-out partition | 86 | 20% stratified; used for classification metrics |
| Cleaned-text overlap | 0 | Expected to be zero |
| Labels in confusion matrix | 36 semantic intents plus fallback outcome | Fallback is an outcome, not a training intent |

Canonicalisation note  The source file contains 432 raw phrases. Evaluation removed 4 same-label duplicate cleaned row(s), leaving 428 unique canonical examples. This is an intentional preprocessing control, not a stale-snapshot mismatch.

### Class balance and source curation

The current intent inventory ranges from 3 to 29 phrases per class, with a median of 12.0. This imbalance, combined with closely related labels, motivates weighted metrics and class_weight='balanced'. New phrases should be added where error analysis shows a real lexical gap, then checked for duplicate cleaned text and label ambiguity before evaluation.

The dataset is stored locally as JSON with one tag, multiple user-phrase patterns, and one or more controlled responses per intent. Topics and factual response anchors are checked against official TAR UMT pages listed in Section 5.5 and data/response_quality_test.json. The project does not claim that the phrase set is a scraped public benchmark or a big-data corpus.

*Figure 2: Distribution of raw training phrases across the current intent inventory*

## 3.4 Preprocessing and feature extraction

Convert the query to lowercase.

Remove bracketed text, URLs, HTML fragments, punctuation, and line breaks; preserve digits and alphanumeric course codes because years, fees, and codes carry meaning.

Split on whitespace and apply WordNet lemmatisation with a safe fallback when the resource is unavailable.

Transform cleaned text into sublinear character-boundary TF-IDF features with analyzer = 'char_wb' and ngram_range = (3, 5), representing within-word fragments while respecting word boundaries.

Retain stop words because character features and question words such as where, when, and how can distinguish university intents.

The preprocessing is deterministic and shared by the local classifier and local simulator. It is not asserted to be part of Google Dialogflow's cloud execution path.

## 3.5 Algorithms and member approaches

### 3.5.1 Member 1: Dialogflow ES configuration and local evaluation surrogate

Member 1's platform approach is represented by Dialogflow ES configuration artifacts, including intents, training phrases, four custom entities, eight structured parameters, entity annotations, required-slot prompting, controlled static responses, and welcome/fallback handling. Webhook fulfillment is disabled by design, avoiding an external deployment dependency during the assessed demonstration. The repository also contains DialogflowSimulatorClient, a transparent local pattern/Jaccard implementation used solely for offline, train-only evaluation. This simulator is reproducible but is not Google's model and provides no evidence of cloud accuracy.

### 3.5.2 Member 2: TF-IDF and Logistic Regression

Member 2's deployed local classifier uses TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True) followed by LogisticRegression(C=30.0, max_iter=2000, random_state=42, class_weight='balanced'). The character-boundary representation captures within-word fragments and is more tolerant of minor spelling and inflection changes than the earlier word-only configuration. Exact cleaned training phrases receive confidence 1.0. Otherwise, the maximum class probability is compared with a confidence threshold of 0.20. Predictions below the threshold return fallback; during normal operation they may be logged to data/unrecognized_queries.json for human review. Evaluation disables logging so that the test procedure does not alter operational data.

### 3.5.3 Model-selection justification

*Table 6: Three-fold out-of-fold model-selection comparison*

| Candidate | Accuracy at 0.20 | Weighted F1 at 0.20 | Coverage at 0.20 | Decision |
| --- | --- | --- | --- | --- |
| Word TF-IDF (1–2) + LR C=10 | 0.6636 | 0.7351 | 0.7407 | Not selected |
| Character-boundary TF-IDF (3–5) + LR C=30 | 0.7874 | 0.8176 | 0.9019 | Selected |
| Word/character feature union + LR C=30 | 0.7710 | 0.8022 | 0.8902 | Not selected |

Candidate selection used 3-fold stratified out-of-fold predictions with random_state = 42 and the recorded selection metric, deployment-threshold weighted F1. The selected character-boundary candidate records 0.8176 weighted F1 and 0.9019 coverage at threshold 0.20, improving the thresholded word-only candidate while remaining simpler than a feature union. Because the out-of-fold comparison covers the same canonical corpus used for subsequent development evaluation, the final held-out numbers should be treated as model-development evidence rather than a sealed external test.

### 3.5.4 Baselines

Multinomial Naïve Bayes with alpha = 1.0 and a unigram TF-IDF representation provides a probabilistic lexical baseline. Linear SVM with C = 0.1, max_iter = 1000, random_state = 42, and unigram TF-IDF provides a margin-based baseline. The baselines do not generate responses and are therefore N/A for BLEU/ROUGE.

## 3.6 Evaluation protocol and metrics

*Table 7: Evaluation protocol*

| Control | Recorded setting | Rationale |
| --- | --- | --- |
| Split | 80/20 stratified | Preserve class proportions where possible |
| Random seed | 42 | Repeatable partition |
| Training/test counts | 342 / 86 | Make sample size explicit |
| Text leakage check | 0 overlaps | Prevent cleaned duplicates across partitions |
| Member 1 scope | local simulator; training phrases only; not cloud | Prevent cloud-performance mislabelling |
| Member 2 threshold | 0.2 | Match the deployed confidence policy |
| Averaging | Weighted precision, recall, and F1 | Account for class support |

Accuracy is the proportion of held-out queries assigned the expected label. Weighted precision measures the reliability of predicted labels, weighted recall measures recovered instances, and weighted F1 is their harmonic balance by class support. Coverage is the proportion receiving a non-fallback label; fallback rate is its complement for the thresholded systems.

For response quality, the protocol uses human-authored reference answers anchored to official TAR UMT sources. Corpus BLEU and mean ROUGE-1 F1 may be computed only after those references are independent of the candidate-selection templates. The repository currently contains 14 reference case(s), and the report uses only scores present in evaluation_results.json. The current evaluation artifact records the resulting scores for the member response engines; they are interpreted only as lexical-overlap evidence and not as proof of factual correctness.

For usability, the Google Form uses five Likert statements rated from 1 (strongly disagree) to 5 (strongly agree). The verified artifact contains 5 anonymous responses collected from 12 to 24 August 2026. The report presents item means, medians, and the predeclared favorable rate (ratings of 4 or 5). No inferential or representative claim is made because the sample is small.

## 3.7 Validity, ethics, and data governance

- Construct validity: report weighted metrics with fallback behaviour; do not equate text overlap with factual correctness.

- Internal validity: check cleaned train/test overlap and restrict every local evaluator to training phrases.

- External validity: 86 held-out examples across many intents do not establish production performance.

- Cloud validity: the offline simulator does not validate Dialogflow ES; authentic console/API tests are required.

- Privacy: minimise survey fields and operational logs, redact identifiers, and control access.

- Factual currency: point time-sensitive answers to official sources and review them periodically.

# 4. Results and Discussion

## 4.1 Classification results

*Table 8: Leakage-free held-out classification metrics from evaluation_results.json*

| Model | Engine/evaluation scope | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Member 1 (Dialogflow ES) | Local train-only Dialogflow-style simulator (not cloud) | 0.6163 | 0.6238 | 0.6163 | 0.5983 |
| Member 2 (TF-IDF + Logistic Reg) | Deployed local ML; confidence threshold=0.20 | 0.7442 | 0.8275 | 0.7442 | 0.7626 |
| Baseline 1 (Multinomial Naïve Bayes) | Local probabilistic baseline | 0.4651 | 0.5369 | 0.4651 | 0.4419 |
| Baseline 2 (Linear SVM) | Local support-vector baseline | 0.7674 | 0.7651 | 0.7674 | 0.7459 |

Source: data/evaluation_results.json generated at 2026-08-24T18:02:56.633862. All classification metrics use the same recorded held-out split.

*Figure 3: Held-out intent-classification performance by model*

The strongest recorded accuracy is 0.7674 for Baseline 2 (Linear SVM). Member 2's local deployment configuration reaches 0.7442 accuracy and 0.7626 weighted F1. Member 1's local simulator records 0.6163 accuracy and 0.5983 weighted F1; this low score belongs to the train-only simulator and must not be re-labelled as Dialogflow cloud performance.

The result ordering suggests that a margin-based linear model currently generalises better than the other tested lexical approaches on the held-out phrases. Member 2's classifier is affected by both multiclass separation and its confidence gate. The comparison does not establish that a particular algorithm is universally superior; it reflects this split, preprocessing, feature configuration, and small class supports.

## 4.2 Coverage, fallback, and response-quality status

*Table 9: Coverage, fallback, and response-quality evidence*

| Model | Coverage | Fallback | Text-overlap metrics | Evidence status |
| --- | --- | --- | --- | --- |
| Member 1 (Dialogflow ES) | 0.9186 | 0.0814 | BLEU 0.1460 ROUGE-1 0.4086 | Independent reference test (14 cases); intent accuracy=0.7143; coverage=1.0000 |
| Member 2 (TF-IDF + Logistic Reg) | 0.8721 | 0.1279 | BLEU 0.1689 ROUGE-1 0.4137 | Independent reference test (14 cases); intent accuracy=0.7857; coverage=0.9286 |
| Baseline 1 (Multinomial Naïve Bayes) | 1.0000 | 0.0000 | BLEU N/A ROUGE-1 N/A | N/A - no independent human-authored response reference set was provided |
| Baseline 2 (Linear SVM) | 1.0000 | 0.0000 | BLEU N/A ROUGE-1 N/A | N/A - no independent human-authored response reference set was provided |

*Figure 4: Coverage and fallback behaviour on the held-out set*

Member 2 answers 87.21% of the held-out cases and falls back on 12.79%. A high fallback rate can be operationally safer than an unsupported confident answer. This remains a material usability cost that should be examined during threshold tuning. Threshold tuning must therefore be evaluated as a coverage–accuracy trade-off rather than optimised for one metric alone.

### 4.2.1 Authentic end-to-end prototype evidence

*Figure 5: Authentic Streamlit end-to-end example using the Member 2 local classifier*

This implementation artifact (file timestamp 2026-08-24T18:07:26+08:00) confirms that the final local route can accept a natural-language query, expose the active Member 2 engine, return the fees intent with confidence 0.9761, and provide official Malaysian and international fee-guide links. It is one successful demonstration case, not an estimate of overall accuracy; general performance is reported only from the held-out evaluation above.

### 4.2.2 Labelled end-to-end smoke probes

*Table 10: Labelled local smoke-probe summary*

| Artifact | Queries | Member 2 local model | Member 1 local simulator | Timestamp |
| --- | --- | --- | --- | --- |
| data/latest_test_results.json | 10 | 10/10 labelled probes passed | 10/10 labelled probes passed | 2026-08-24 20:09:09 |

These manually labelled probes are deterministic end-to-end smoke checks over selected queries. The column labelled Member 1 refers to the repository's local Dialogflow-style simulator, not the Google Dialogflow ES cloud service. A 10/10 smoke result verifies the tested cases only and is not a substitute for the held-out classification results in Table 8.

## 4.3 Confusion and error analysis

*Figure 6: Most frequent off-diagonal confusion pairs for the two member approaches*

*Table 11: Error-analysis summary derived from recorded test cases*

| Model | Leading off-diagonal pairs | Fallback predictions |
| --- | --- | --- |
| Member 1 (Dialogflow ES) | abusive_language → fallback (2); greeting → fallback (2); accommodation → clubs_societies (1); accommodation → facilities (1) | 7 |
| Member 2 (TF-IDF + Logistic Reg) | contact → bot_identity (2); greeting → fallback (2); abusive_language → assessment_rules (1); abusive_language → fallback (1) | 11 |

Several errors are semantically plausible: location, campus map, contact, facilities, sport, intake, semester break, and examination questions share vocabulary. Short social utterances such as greetings, thanks, identity questions, and affection also offer few discriminating terms. These patterns imply that improvement should begin with label-boundary review and targeted paraphrase collection rather than indiscriminate duplication of existing phrases.

The confusion chart counts are small because the held-out set is distributed across many labels. A difference of one example can materially change a class score. Per-class support and repeated cross-validation would be needed before making stronger comparative claims.

## 4.4 User satisfaction and usability

Five anonymous Google Forms responses were collected between 12 and 24 August 2026 using the exact five-item questionnaire reproduced in Appendix C. All complete responses were retained, including one respondent who selected 1 for every item. Across all 25 ratings, the mean was 3.76/5, the median was 4.0, and 17 ratings (68.0%) were favorable. Favorable was defined before analysis as a rating of 4 or 5.

*Table 12: Verified user-satisfaction results (N=5)*

| Survey item | Mean / 5 | Median | Favorable (4-5) |
| --- | --- | --- | --- |
| Q1. Intent understanding | 4.00 | 5.0 | 4/5 (80%) |
| Q2. Answer clarity and relevance | 4.00 | 5.0 | 4/5 (80%) |
| Q3. Interface usability | 3.80 | 5.0 | 3/5 (60%) |
| Q4. Response speed | 3.60 | 4.0 | 3/5 (60%) |
| Q5. Overall satisfaction | 3.40 | 4.0 | 3/5 (60%) |

*Figure 7: Verified five-item usability survey results*

Intent understanding and answer clarity received the strongest item means (4.00/5; 80% favorable each). Interface usability averaged 3.80/5, while response speed averaged 3.60/5. Overall satisfaction was the lowest-rated item at 3.40/5, with three of five respondents rating it 4 or 5. The result suggests that the prototype is generally understandable but still needs reliability, latency, and overall experience improvements. Because N=5 and recruitment was not probabilistic, these findings describe only this pilot group and should not be generalized to the wider student population.

## 4.5 Objective-by-objective interpretation

*Table 13: Objective attainment*

| Objective | Finding | Status |
| --- | --- | --- |
| FAQ coverage | 36 semantic intents in the current source inventory | Met at prototype scope |
| Two approaches | Dialogflow configuration track and local ML track are documented separately | Met; contribution evidence to confirm |
| Local implementation | Character-boundary TF-IDF (3–5), LR C=30/max_iter=2000/balanced, threshold=0.20 | Met |
| Leakage-free comparison | Recorded overlap=0; four model rows | Met for local evaluation |
| Cloud Dialogflow performance | No authentic cloud confusion matrix or API test artifact | Not yet demonstrated |
| Response-quality evidence | Independent reference test (14 cases); intent accuracy=0.7857; coverage=0.9286 | Measured |
| User satisfaction | N=5; five-item mean 3.76/5; favorable 68.0% | Measured descriptively |

## 4.6 Limitations and practical implications

- The evaluation snapshot is small relative to the number of intents and may be stale when intents.json changes.

- The Member 1 offline score measures a local pattern/Jaccard surrogate, not Dialogflow ES cloud NLU.

- The member approaches are not symmetric: Member 2 includes a confidence gate, while the baselines always classify; coverage must accompany accuracy.

- Template responses reduce hallucination but can become outdated and do not support rich multi-turn context.

- Automatic text-overlap metrics are available from the independent-reference instrument but still require human factuality and usability review.

- The verified usability pilot has only five respondents, so it supports descriptive feedback but not statistical generalisation.

Practically, the strongest next step is not to advertise the current prototype as accurate. It is to review the intent taxonomy, add independently phrased examples to weak labels, tune the confidence threshold on a validation set, rerun the leakage-free evaluation, and capture authentic Dialogflow test evidence. This sequence directly targets the observed errors and evidence gaps.

# 5. Conclusion, References and Sources

## 5.1 Achievements

The project has produced an end-to-end university FAQ chatbot prototype with a clear task scenario, a structured intent/response inventory, a Dialogflow ES configuration path, an offline Python classifier, a Streamlit interface, controlled fallback handling, test code, a reproducible evaluation artifact, and a verified five-response usability pilot. The documentation separates cloud configuration from the local simulator and keeps every reported metric traceable to a recorded artifact.

The present results establish a transparent baseline rather than a production claim. Baseline 2 (Linear SVM) has the highest recorded accuracy (0.7674), while Member 2's deployed configuration records 0.7442 accuracy with 0.8721 coverage. These figures provide a concrete starting point for error-driven improvement.

## 5.2 Limitations

The current system is single-turn, English-first, dependent on a small project-specific phrase set, and limited to predefined responses. Dataset and evaluation snapshots are not automatically synchronised unless evaluate.py and this builder are rerun in sequence. Dialogflow cloud accuracy has not been demonstrated by the offline surrogate. The usability survey is genuine but preliminary because it contains only five anonymous respondents. Response-overlap scores are available, but they do not establish factual correctness or usability.

## 5.3 Future work

Resolve overlapping intent definitions and expand weak classes with genuinely new paraphrases collected under a documented protocol.

Use a validation partition or nested cross-validation to tune the 0.20 threshold without optimising on the final test set.

Evaluate class-balanced metrics, per-class support, calibration, latency, and repeated splits in addition to the current weighted metrics.

Run an authentic Dialogflow ES console/API test on the same locked queries and preserve dated screenshots or JSON responses.

Expand and review the independent-source response test, then pair BLEU/ROUGE with blinded human ratings for factuality, relevance, clarity, and source usefulness.

Repeat the Appendix C survey with a larger and more diverse voluntary sample, preserve anonymous responses, and compare results with the current N=5 pilot without discarding unfavorable ratings.

Add multi-turn context and a source-aware retrieval layer only after factuality, privacy, and update governance are defined.

## 5.4 References

Dibitonto, M., Leszczynska, K., Tazzi, F., & Medaglia, C. M. (2018). Chatbot in a campus environment: Design of LiSA, a virtual assistant to help students in their university life. In M. Kurosu (Ed.), Human–computer interaction. Interaction technologies (Lecture Notes in Computer Science, Vol. 10903, pp. 103–116). Springer. https://doi.org/10.1007/978-3-319-91250-9_9

Google Cloud. (n.d.). Dialogflow ES documentation. https://cloud.google.com/dialogflow/es/docs

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. In Text summarization branches out (pp. 74–81). Association for Computational Linguistics. https://aclanthology.org/W04-1013/

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (pp. 311–318). Association for Computational Linguistics. https://doi.org/10.3115/1073083.1073135

Pérez-Soler, S., Juárez-Puerta, S., Guerra, E., & de Lara, J. (2021). Choosing a chatbot development tool. IEEE Software, 38(4), 94–103. https://doi.org/10.1109/MS.2020.3030198

Ranoliya, B. R., Raghuwanshi, N., & Singh, S. (2017). Chatbot for university related FAQs. In 2017 International Conference on Advances in Computing, Communications and Informatics (ICACCI) (pp. 1525–1530). IEEE. https://doi.org/10.1109/ICACCI.2017.8126057

## 5.5 Dataset, factual-source, and tool acknowledgement

The intent phrases and response templates are stored in data/intents.json and are treated as a team-curated project dataset, not a public big-data benchmark. Official TAR UMT URLs are used as factual anchors for the independent response-quality instrument. Complete machine-readable case-to-source mappings are preserved in data/response_quality_test.json.

*Table 14: Official factual-source inventory used by the response-quality instrument*

| Topic | Official URL | Repository provenance |
| --- | --- | --- |
| Location | https://www.tarc.edu.my/contact-us/ | response_quality_test.json |
| Intake | https://www.tarc.edu.my/admissions/new-student/academic-calendar/ | response_quality_test.json |
| Fees | https://www.tarc.edu.my/apply-and-study/fees/malaysians-student-fees/ | response_quality_test.json |
| Fees | https://www.tarc.edu.my/intfees/international-student-fees-guide-for-year-2026-intakes/ | response_quality_test.json |
| Admission Documents | https://www.tarc.edu.my/admissions/application-and-enrolment-status-enquiry/ | response_quality_test.json |
| Accommodation | https://www.tarc.edu.my/dsa/content.jsp?cat_id=1210BF6A-BE89-49AC-B823-DC729888CB71&fmenuid=DCF6BB08-61CA-4FB4-9057-18913B3987C4 | response_quality_test.json |
| Accommodation | https://www.tarc.edu.my/admissions/faqs/ | response_quality_test.json |
| Library | https://library.tarc.edu.my/ | response_quality_test.json |
| Careers | https://scdcstu.tarc.edu.my/ | response_quality_test.json |
| Scholarship | https://www.tarc.edu.my/dsa/financial-aid/financial-aid/ | response_quality_test.json |
| Clubs Societies | https://www.tarc.edu.my/dsa/ | response_quality_test.json |
| Sports | https://www.tarc.edu.my/files/dsa/7BE77EC5-DA21-4AD7-92AE-3717934D5004.pdf | response_quality_test.json |
| Transport | https://www.tarc.edu.my/dsa/a/transportation/university-college-bus-service/ | response_quality_test.json |
| Assessment Rules | https://examination.tarc.edu.my/examination-services/faqs | response_quality_test.json |
| Setup/Documentation Source | https://cloud.google.com/dialogflow/es/docs/agents-settings | docs/dialogflow_setup.md |

*Table 15: Software and artifact acknowledgement*

| Component | Recorded use | Repository evidence |
| --- | --- | --- |
| Python | Application, model selection, evaluation, and document builder | app.py; model_selection.py; evaluate.py; tools/build_final_report.py |
| NLTK | Lemmatisation and optional BLEU token processing | src/preprocessing.py; evaluate.py |
| scikit-learn | TF-IDF, Logistic Regression, Naïve Bayes, SVM, metrics, split, cross-validation | src/ml_model.py; model_selection.py; evaluate.py; data/model_selection_results.json |
| Streamlit | Interactive prototype interface | app.py |
| Dialogflow ES | Platform configuration, entities, parameter prompting, controlled static responses, and fallback handling | dialogflow_agent.zip; data/entities.json; docs/dialogflow_setup.md |
| Repository data | Intent patterns, feedback, tests, evaluation results | data/ directory |

# Appendix A. Contribution record and prototype evidence

Member verification required  The responsibilities below reflect the repository's two-track structure. Both students must confirm the allocation and add dated evidence before submission; the report does not infer authorship from file names alone.

*Table A1: Individual contribution record requiring member confirmation*

| Member | Proposed responsibility record | Evidence to attach | Confirmation |
| --- | --- | --- | --- |
| Member 1 [TO BE PROVIDED] | Dialogflow ES intent/entity/response configuration; parameter annotations and required-slot prompting; exported agent artifact; train-only simulator integration | Export timestamp; parameter-prompt test; authentic console/API test; meeting log | Member initials/date: __________ |
| Member 2 [TO BE PROVIDED] | NLTK preprocessing; character-boundary TF-IDF (3–5) + balanced Logistic Regression (C=30) pipeline; confidence gate; model selection; local evaluation and UI integration | Commit/version history; model-selection artifact; test output; code walkthrough; meeting log | Member initials/date: __________ |
| Shared | Problem framing; dataset review; literature synthesis; error analysis; final QA and demonstration | Minutes, review notes, rehearsal checklist | Both initials/date: __________ |

## Prototype demonstration evidence checklist

- ☐ Show a clean installation and launch of the Streamlit application.

- ☐ Demonstrate at least one correct local prediction, one ambiguous query, and one confidence-triggered fallback.

- ☐ Open the fallback log and explain human review without exposing personal data.

- ☐ Show the Dialogflow ES agent in the genuine console and test locked queries; retain dated screenshots or exported JSON.

- ☐ Run evaluate.py and explain the split, leakage check, metrics, coverage, and confusion errors.

- ☐ Each member presents and answers questions about their own implementation.

Screenshot policy  Insert only screenshots captured from the real local application or authenticated Dialogflow console. No synthetic console image is included in this report. Record the capture date and a short description under each inserted figure.

# Appendix B. Reproducibility checklist

After activating the project virtual environment, run the portable commands below from the repository root. Preserve the listed repository artifacts and check each item after a clean rerun.

*Table B1: Reproducibility and submission checklist*

| Done | Area | Verification action |
| --- | --- | --- |
| ☐ | Environment | Install requirements.txt; record Python and package versions. |
| ☐ | Dataset | Verify data/intents.json validates and record its phrase/intent counts. |
| ☐ | Response references | Validate data/response_quality_test.json sources and independence. |
| ☐ | Model selection | Run model_selection.py; confirm the three-fold OOF artifact and selected character-boundary candidate. |
| ☐ | Evaluation | Run evaluate.py; confirm seed 42, stratification, and zero cleaned-text overlap. |
| ☐ | Snapshot consistency | Confirm evaluation_results.json was generated after the final intents.json edit. |
| ☐ | Tests | Run the full automated test suite and retain the terminal output. |
| ☐ | Report | Run the command below to refresh charts and AI Report - Final.docx. |
| ☐ | Visual QA | Render the DOCX with canonical render_docx.py and inspect every page. |
| ☐ | Cloud evidence | Export the Dialogflow agent and capture authentic locked-query results. |
| ☐ | Survey | Preserve data/user_feedback_verified.json; confirm N=5 and collection dates before rebuilding. |

## Regeneration commands

python model_selection.py

python evaluate.py

python -m unittest discover -s tests -v

python src/create_dialogflow_zip.py

python tools/build_final_report.py

Evaluation artifact: evaluation_results.json generated at 2026-08-24T18:02:56.633862. Builder output generated at 2026-08-24T20:25:35+08:00.

# Appendix C. Verified user-satisfaction instrument and data

Observed pilot  The frozen anonymized Google Forms snapshot contains N=5 complete responses collected from 12 to 24 August 2026. All five complete responses are retained, including the all-1 response. Results are descriptive and are not generalized beyond this small pilot.

## Questionnaire and scoring rule

Scale: 1 = strongly disagree, 2 = disagree, 3 = neither agree nor disagree, 4 = agree, and 5 = strongly agree. A favorable response is defined as 4 or 5. The questionnaire collected ratings and timestamps only; the exported research snapshot contains no names or student IDs.

*Table C1: Exact five-item Google Forms questionnaire*

| Item | Statement |
| --- | --- |
| Q1 | The chatbot correctly understands student inquiry questions and intent. |
| Q2 | The chatbot's answers are clear, informative, and relevant to university procedures. |
| Q3 | The web user interface (Streamlit GUI) is easy to navigate and chat with. |
| Q4 | The chatbot responds promptly without noticeable delay. |
| Q5 | Overall, I am satisfied with the automated university inquiry chatbot system. |

## Item-level descriptive results

*Table C2: Verified descriptive statistics*

| Item | Mean | Median | Favorable |
| --- | --- | --- | --- |
| Q1 | 4.00/5 | 5.0 | 4/5 (80%) |
| Q2 | 4.00/5 | 5.0 | 4/5 (80%) |
| Q3 | 3.80/5 | 5.0 | 3/5 (60%) |
| Q4 | 3.60/5 | 4.0 | 3/5 (60%) |
| Q5 | 3.40/5 | 4.0 | 3/5 (60%) |
| All 25 ratings | 3.76/5 | 4.0 | 17/25 (68.0%) |

## Anonymized response matrix

*Table C3: Retained complete responses*

| Response | Date | Q1 | Q2 | Q3 | Q4 | Q5 |
| --- | --- | --- | --- | --- | --- | --- |
| GF-001 | 2026-08-12 | 5 | 5 | 3 | 4 | 3 |
| GF-002 | 2026-08-24 | 1 | 1 | 1 | 1 | 1 |
| GF-003 | 2026-08-24 | 5 | 5 | 5 | 5 | 5 |
| GF-004 | 2026-08-24 | 4 | 4 | 5 | 3 | 4 |
| GF-005 | 2026-08-24 | 5 | 5 | 5 | 5 | 4 |

## Data-quality decisions and limitations

- The separate local UI demo submission was excluded because it was not part of the Google Forms study.

- No complete Google Forms response was removed; an unfavorable response is not an exclusion criterion.

- Only complete integer ratings from 1 to 5 were accepted by the verified-data validator.

- The small convenience sample (N=5) is suitable for pilot feedback but not inferential statistics or population claims.

- The reproducible anonymized snapshot is stored in data/user_feedback_verified.json; the live response sheet should remain access-controlled.

# Appendix D. Authentic evidence register

This register distinguishes captured local evidence from artifacts still required. Cloud rows remain pending because a local simulator or synthetic image cannot validate Dialogflow ES.

*Table D1: Evidence register for final submission*

| Evidence ID | Required artifact | Minimum metadata | Status |
| --- | --- | --- | --- |
| E1 | Streamlit home/interface screenshot | File timestamp 2026-08-24T18:07:26+08:00; Member 2 selection visible | CAPTURED |
| E2 | Local correct-intent interaction | Tuition-fee query; fees intent; confidence 0.9761; official-source response | CAPTURED |
| E3 | Local fallback and review log | Query; threshold; safe redaction; timestamp | TO BE CAPTURED |
| E4 | Dialogflow ES intents view | Authenticated console; agent name; capture date | TO BE CAPTURED |
| E5 | Dialogflow locked-query test | Query set ID; intent; confidence; API/console provenance | TO BE CAPTURED |
| E6 | Evaluation/test terminal output | Command; timestamp; test count; pass/fail | TO BE CAPTURED |
| E7 | Verified Google Forms survey snapshot | N=5; 12-24 August 2026; anonymized response matrix | CAPTURED |

E1/E2 artifact: report_assets/app_chatbot_e2e.png. Figure 5 reproduces it without alteration.

Do not substitute  The local Dialogflow-style simulator is not an acceptable substitute for E4 or E5. E7 uses the verified anonymized Google Forms snapshot; local demo feedback is excluded.

# Appendix E. Plagiarism Statement Form — Member 1

Complete this form personally. The document builder intentionally leaves all identity, signature, and date fields blank.

*Table E1: Member 1 identification*

| Field | To be completed by the student |
| --- | --- |
| Name | [TO BE PROVIDED] |
| Student ID | [TO BE PROVIDED] |
| Course code | BMCS2003 |
| Assignment title | TAR UMT University Inquiry Chatbot |
| Tutorial group | [TO BE PROVIDED] |

## Declaration

I declare that the work submitted for this assignment is my own contribution except where sources and team contributions are clearly acknowledged. I have not copied another student's or group's work, shared material in a way that enables academic misconduct, fabricated research participants or results, or represented generated or third-party material as my own without appropriate acknowledgement. I understand that I remain responsible for checking the accuracy, originality, citations, data provenance, and submitted code.

- ☐ I have reviewed the final report and source code.

- ☐ I have checked that my individual contribution record is accurate.

- ☐ I have checked that all reported metrics are traceable to evaluation_results.json, user_feedback_verified.json, or clearly labelled N/A.

- ☐ I have complied with TAR UMT academic-integrity requirements and the instructions supplied for this assignment.

Student signature: ______________________________________________

Date: __________________________________________________________

Witness / tutor (if required): ____________________________________

Unsigned by design  No name, signature, date, or consent is inserted automatically. The student must complete and sign this form after reviewing the final submission.

# Appendix F. Plagiarism Statement Form — Member 2

Complete this form personally. The document builder intentionally leaves all identity, signature, and date fields blank.

*Table F1: Member 2 identification*

| Field | To be completed by the student |
| --- | --- |
| Name | [TO BE PROVIDED] |
| Student ID | [TO BE PROVIDED] |
| Course code | BMCS2003 |
| Assignment title | TAR UMT University Inquiry Chatbot |
| Tutorial group | [TO BE PROVIDED] |

## Declaration

I declare that the work submitted for this assignment is my own contribution except where sources and team contributions are clearly acknowledged. I have not copied another student's or group's work, shared material in a way that enables academic misconduct, fabricated research participants or results, or represented generated or third-party material as my own without appropriate acknowledgement. I understand that I remain responsible for checking the accuracy, originality, citations, data provenance, and submitted code.

- ☐ I have reviewed the final report and source code.

- ☐ I have checked that my individual contribution record is accurate.

- ☐ I have checked that all reported metrics are traceable to evaluation_results.json, user_feedback_verified.json, or clearly labelled N/A.

- ☐ I have complied with TAR UMT academic-integrity requirements and the instructions supplied for this assignment.

Student signature: ______________________________________________

Date: __________________________________________________________

Witness / tutor (if required): ____________________________________

Unsigned by design  No name, signature, date, or consent is inserted automatically. The student must complete and sign this form after reviewing the final submission.
