import os
import json
import tempfile
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ml_model import UniversityIntentClassifier
from src.preprocessing import analyze_sentiment
from evaluate import evaluate_models, get_user_satisfaction_metrics, save_user_feedback

# Page Config
st.set_page_config(
    page_title="University Inquiry Chatbot - TARUMT AI Assignment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and fixed sticky header
st.markdown("""
<style>
    /* Reduce top padding of main container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Sticky Top Header Banner */
    .sticky-header-container {
        position: sticky;
        top: 0px;
        z-index: 9999;
        background-color: var(--background-color, rgba(255, 255, 255, 0.95));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 0.6rem 0 0.4rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(226, 232, 240, 0.8);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 0.4rem;
    }

    /* Sticky Navigation Tabs Bar */
    div[data-testid="stTabs"] > div[role="tablist"] {
        position: sticky;
        top: 5.2rem;
        z-index: 9998;
        background-color: var(--background-color, rgba(255, 255, 255, 0.95));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(226, 232, 240, 0.5);
    }

    .badge-dialogflow {
        background-color: #4285F4;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-ml {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-dnn {
        background-color: #8B5CF6;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""",unsafe_allow_html=True)

# Initialize Session State
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "intents.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "unrecognized_queries.json")


def load_json_file(path, default):
    """Load a JSON file without crashing the demo on an empty/corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_write_json(path, payload):
    """Write JSON atomically so an interrupted admin action cannot corrupt data."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".chatbot-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

@st.cache_resource
def load_ml_classifier():
    clf = UniversityIntentClassifier()
    clf.train(DATASET_PATH)
    return clf

ml_model = load_ml_classifier()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your University Inquiry Assistant. Ask me about courses, fees, hostels, campus facilities, or admission procedures!",
            "model_used": "System",
            "intent": "greeting",
            "confidence": 1.0
        }
    ]

# Sidebar Controls. Keep the mark local so the offline demo never shows a broken image.
st.sidebar.markdown(
    "<div style='font-size:3.25rem; line-height:1; margin:0.25rem 0 0.75rem'>🏫</div>",
    unsafe_allow_html=True,
)
st.sidebar.title("📌 Assignment Control")
st.sidebar.markdown("**Course**: Artificial Intelligence  \n**Topic**: 5. Chatbot Development  \n**Title**: University Inquiry Chatbot  \n**Team Size**: 2 Members")

st.sidebar.divider()
st.sidebar.subheader("🤖 Active Model Selection")
active_model = st.sidebar.radio(
    "Choose Solution to Test:",
    options=[
        "Member 1: Google Dialogflow (Platform)",
        "Member 2: Python ML (TF-IDF + Logistic Regression)"
    ],
    index=1
)

st.sidebar.divider()
if st.sidebar.button("🔄 Reload Model & Dataset Cache"):
    st.cache_resource.clear()
    st.sidebar.success("✅ Model re-trained & memory refreshed!")
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📊 Group Member Roles")
st.sidebar.markdown("""
- **Member 1**: Dialogflow Platform Agent (Option 2)
- **Member 2**: Python ML Classifier Pipeline (Option 1)
- **Evaluation**: held-out intent metrics, response-quality evidence, and user feedback
""")

# Fixed Sticky Header
st.markdown("""
<div class='sticky-header-container'>
    <div class='main-title'>🎓 University Inquiry Chatbot</div>
    <div class='sub-title'>AI Assistant for Student Inquiries, Course Info & Campus Services</div>
</div>
""", unsafe_allow_html=True)

# Main Content Layout Tabs
tab_chat, tab_metrics, tab_analytics, tab_learning, tab_info = st.tabs([
    "💬 Chatbot Demo",
    "📈 Model Performance Dashboard",
    "📊 Inquiry Data Analytics",
    "🔄 Active Learning Review",
    "📄 Assignment Spec & Schema"
])

# Helper for ChatGPT-style word streaming effect
def stream_response(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

with tab_chat:

    # Status Badge
    if "Dialogflow" in active_model:
        st.markdown("<span class='badge-dialogflow'>Member 1 - Google Dialogflow ES Messenger</span>", unsafe_allow_html=True)
        # Embed Live Dialogflow Messenger Component
        st.markdown("#### 💬 Embedded Google Dialogflow Agent")
        st.caption("The widget requires internet access and an enabled Dialogflow agent. Offline benchmark results are reported separately and are not presented as cloud measurements.")
        st.components.v1.html(
            """
            <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
            <style>
              df-messenger {
                --df-messenger-bot-message: #1e293b;
                --df-messenger-button-titlebar-color: #1e88e5;
                --df-messenger-chat-background: #0f172a;
                --df-messenger-font-color: #ffffff;
                --df-messenger-user-message: #2563eb;
                width: 100%;
                height: 480px;
              }
            </style>
            <df-messenger
              intent="WELCOME"
              chat-title="TARUMT University Bot"
              agent-id="f238ed0f-ff40-437d-9cdc-aa7ca4408f0f"
              language-code="en"
            ></df-messenger>
            """,
            height=500
        )
    else:
        st.markdown("<span class='badge-ml'>Active Engine: Member 2 - Python Machine Learning (TF-IDF + Logistic Regression)</span>", unsafe_allow_html=True)

        # ChatGPT-Style Fixed Scrollable Chat Container
        chat_box = st.container(height=480)

        # Display Chat History inside scrollable container
        with chat_box:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "intent" in message and message["intent"] != "greeting":
                        sentiment_info = message.get("sentiment", {"sentiment": "Neutral", "emoji": "😐"})
                        st.caption(f"🎯 **Intent**: `{message['intent']}` | ⚡ **Confidence**: `{message['confidence']}` | {sentiment_info['emoji']} **Sentiment**: `{sentiment_info['sentiment']}`")

        # Quick Question Suggestion Chips
        st.caption("💡 **Quick FAQ Prompts**: Click to test popular inquiries:")
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        selected_prompt = None
        with btn_c1:
            if st.button("📚 Courses Offered?", key="q_course"):
                selected_prompt = "What courses are offered?"
        with btn_c2:
            if st.button("💰 Tuition Fees?", key="q_fees"):
                selected_prompt = "How much are the tuition fees?"
        with btn_c3:
            if st.button("🏠 Hostel Info?", key="q_hostel"):
                selected_prompt = "Tell me about hostel accommodation"
        with btn_c4:
            if st.button("📝 Admission Requirements?", key="q_adm"):
                selected_prompt = "What are the admission requirements?"

        # Chat Input Box (Handles typed input or quick chip click)
        prompt_input = st.chat_input("Type your question here (e.g., 'What courses are offered?', 'How much is the fee?')...")
        prompt = selected_prompt or prompt_input

        if prompt:
            # Display user message inside scrollable chat box
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Analyze sentiment
                sent_dict = analyze_sentiment(prompt)

                # This branch is the local ML demo. Dialogflow uses its own widget above.
                with st.chat_message("assistant"):
                    ml_res = ml_model.get_response(prompt)
                    res_text = ml_res["response"]
                    intent_tag = ml_res["predicted_tag"]
                    conf = ml_res["confidence"]
                    engine_label = "Member 2 (Python ML)"

                    # Smooth ChatGPT-style streaming response
                    st.write_stream(stream_response(res_text))
                    st.caption(f"🎯 **Intent**: `{intent_tag}` | ⚡ **Confidence**: `{conf}` | {sent_dict['emoji']} **Sentiment**: `{sent_dict['sentiment']}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res_text,
                        "model_used": engine_label,
                        "intent": intent_tag,
                        "confidence": conf,
                        "sentiment": sent_dict
                    })
            if selected_prompt:
                st.rerun()

with tab_metrics:
    st.header("📊 Leakage-Free Model Evaluation")
    st.markdown("Reports held-out intent metrics (**Accuracy, Precision, Recall, weighted F1, coverage, and fallback rate**). Response-quality scores are shown only when an independent reference-answer set exists.")

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        st.caption("The evaluator removes duplicate/ambiguous cleaned phrases, then creates one reproducible 80/20 stratified split. The local Dialogflow-style baseline receives training phrases only.")
    with col_btn2:
        rerun_benchmark = st.button("🚀 Run Benchmark", width="stretch")

    if rerun_benchmark or "eval_results_df" not in st.session_state:
        with st.spinner("Training and evaluating four local models on the leakage-free held-out split..."):
            eval_results_df, eval_X_test, eval_y_test, eval_ml_preds = evaluate_models(DATASET_PATH)
            st.session_state.eval_results_df = eval_results_df

    results_df = st.session_state.eval_results_df

    st.subheader("1. Held-Out Intent Metrics")
    st.dataframe(results_df, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Member 1 Local Baseline")
        m1_data = results_df[results_df["Member"].str.contains("Member 1")].iloc[0]
        st.metric("F1-Score (g.i)", f"{m1_data['F1-Score']*100:.1f}%")
        st.metric("Coverage", f"{m1_data['Coverage']*100:.1f}%")
        st.metric("Fallback Rate", f"{m1_data['Fallback Rate']*100:.1f}%")
        st.caption("Train-only offline Dialogflow-style simulator; this is not a cloud Dialogflow score.")

    with col2:
        st.subheader("Member 2 (Python ML)")
        m2_data = results_df[results_df["Member"].str.contains("Member 2")].iloc[0]
        st.metric("F1-Score (g.i)", f"{m2_data['F1-Score']*100:.1f}%")
        st.metric("Coverage", f"{m2_data['Coverage']*100:.1f}%")
        st.metric("Fallback Rate", f"{m2_data['Fallback Rate']*100:.1f}%")
        st.caption("Uses the same confidence threshold (0.20) as the deployed local chatbot.")

    # Comparative Chart
    st.subheader("2. Classification and Coverage Comparison")
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    rename_map = {
        "Accuracy": "Accuracy",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1-Score": "F1-Score",
        "Coverage": "Coverage"
    }

    metrics_melted = pd.melt(results_df, id_vars=["Member"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "Coverage"])
    metrics_melted["value"] = pd.to_numeric(metrics_melted["value"], errors="coerce")
    metrics_melted["Metric"] = metrics_melted["variable"].map(rename_map)

    custom_palette = {
        "Member 1 (Dialogflow ES)": "#2563EB",
        "Member 2 (TF-IDF + Logistic Reg)": "#10B981",
        "Baseline 1 (Multinomial Naïve Bayes)": "#F59E0B",
        "Baseline 2 (Linear SVM)": "#8B5CF6"
    }

    sns.barplot(data=metrics_melted, x="Metric", y="value", hue="Member", ax=ax, palette=custom_palette, edgecolor="none")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0.00 - 1.00)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xlabel("Held-Out Evaluation Metric", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title("Leakage-Free Intent Classification Comparison", fontsize=13, fontweight='bold', pad=12)

    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height) and height > 0:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8, fontweight='bold')

    ax.legend(title="Models Evaluated", title_fontsize='9', fontsize='9', loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0')
    sns.despine(top=True, right=True)
    st.pyplot(fig)

    st.subheader("3. Response Relevancy and Quality (Requirement g.ii)")
    st.caption("BLEU and ROUGE-1 F1 are calculated on a separate, source-grounded reference set whose queries and answers do not occur in the training data. Scores are reproducible and are not obtained by comparing a stored response with itself.")
    response_col1, response_col2 = st.columns(2)
    with response_col1:
        st.markdown("**Member 1 local baseline**")
        st.metric("Corpus BLEU", f"{m1_data['BLEU Score (g.ii)']:.4f}")
        st.metric("ROUGE-1 F1", f"{m1_data['ROUGE-1 Score (g.ii)']:.4f}")
        st.metric("Reference-set Intent Accuracy", f"{m1_data['Response Test Intent Accuracy']*100:.1f}%")
    with response_col2:
        st.markdown("**Member 2 local ML**")
        st.metric("Corpus BLEU", f"{m2_data['BLEU Score (g.ii)']:.4f}")
        st.metric("ROUGE-1 F1", f"{m2_data['ROUGE-1 Score (g.ii)']:.4f}")
        st.metric("Reference-set Intent Accuracy", f"{m2_data['Response Test Intent Accuracy']*100:.1f}%")
    st.info("Low lexical-overlap scores are reported as observed, not inflated. The report discusses why source-link responses can be relevant while sharing few exact n-grams with concise reference answers.")

    st.subheader("4. User Satisfaction & Usability Rating (Requirement g.iii)")
    st.info("📋 **Usability Survey**: Use the 5-point questionnaire to collect genuine responses (1 = Strongly Disagree, 5 = Strongly Agree). The table below reports only saved submissions; no synthetic fallback scores are used. [Open the Google Form](https://docs.google.com/forms/d/e/1FAIpQLSf9lTavaRnmdLv7MhYZb6jzqDwLd3sz47IlFqjgPNAFQk2I8w/viewform)")
    st.dataframe(get_user_satisfaction_metrics(), width="stretch")

    with st.expander("⭐ **Submit Live User Satisfaction Feedback & Rating (Demo Interactive Probe)**"):
        if st.session_state.get("submitted_feedback", False):
            st.success("✅ **Feedback Received**: You have already submitted feedback for this session! Thank you for rating our university chatbot.")
        else:
            st.markdown("Try rating the chatbot live during the demo! Your feedback will dynamically update the system metrics.")
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                fb_name = st.text_input("Optional anonymous alias:", value="")
                fb_acc = st.slider("1. Intent Recognition Accuracy & Precision:", 1, 5, 3)
                fb_qual = st.slider("2. Response Relevancy & Quality:", 1, 5, 3)
            with col_fb2:
                fb_ui = st.slider("3. User Interface & Navigability:", 1, 5, 3)
                fb_speed = st.slider("4. Response Speed / Low Latency:", 1, 5, 3)
                fb_sat = st.slider("5. Overall System Satisfaction:", 1, 5, 3)

            fb_comments = st.text_area("Optional Feedback / Comments:", placeholder="Write your comments here...")
            if st.button("🚀 Submit Feedback Rating", width="stretch"):
                save_user_feedback(fb_name, fb_acc, fb_qual, fb_ui, fb_speed, fb_sat, fb_comments)
                st.session_state.submitted_feedback = True
                st.success("✅ Thank you! Your feedback has been saved and metrics updated dynamically!")
                st.rerun()

with tab_analytics:
    st.header("📊 Intent Dataset Analytics")
    st.markdown("This dashboard summarises the small, curated training dataset. It is exploratory analysis, not a big-data claim.")

    with open(DATASET_PATH, 'r') as f:
        data_json = json.load(f)

    intent_counts = []
    for item in data_json['intents']:
        intent_counts.append({
            "Intent Tag": item['tag'],
            "Training Pattern Count": len(item['patterns'])
        })

    df_analytics = pd.DataFrame(intent_counts).sort_values(by="Training Pattern Count", ascending=False)

    st.subheader("1. Inquired University Topic Categories Distribution")

    col_sel1, col_sel2 = st.columns([2, 2])
    with col_sel1:
        top_n = st.slider("Select Number of Topics to Display in Chart:", min_value=5, max_value=len(df_analytics), value=15, step=5)
    with col_sel2:
        st.caption(f"Showing Top **{top_n}** out of total **{len(df_analytics)}** intent topic classes.")

    df_top = df_analytics.head(top_n)

    fig_analytics, ax_analytics = plt.subplots(figsize=(10, max(4.5, top_n * 0.38)))
    sns.barplot(data=df_top, x="Training Pattern Count", y="Intent Tag", hue="Intent Tag", ax=ax_analytics, palette="viridis", legend=False)
    ax_analytics.set_title(f"Top {top_n} University Inquiry Topic Distributions", fontsize=13, fontweight='bold', pad=12)
    ax_analytics.set_xlabel("Number of Questions / Pattern Variations", fontsize=11, fontweight='bold', labelpad=8)
    ax_analytics.set_ylabel("Intent Tag", fontsize=11, fontweight='bold', labelpad=8)

    for p in ax_analytics.patches:
        width = p.get_width()
        if not pd.isna(width) and width > 0:
            ax_analytics.annotate(f"{int(width)} patterns", (width, p.get_y() + p.get_height() / 2.),
                                  ha='left', va='center', xytext=(6, 0), textcoords='offset points', fontsize=9, fontweight='bold', color='#334155')

    ax_analytics.set_xlim(0, df_top["Training Pattern Count"].max() * 1.20)
    sns.despine(top=True, right=True)
    st.pyplot(fig_analytics)

    st.subheader(f"2. Full Dataset Intent Inventory ({len(df_analytics)} Classes)")
    st.dataframe(df_analytics, width="stretch")

with tab_learning:
    st.header("🔄 Active Learning Review")
    st.markdown("Low-confidence local-model queries are logged for a human reviewer. Any dataset change requires an administrator PIN and explicit confirmation.")

    if os.path.exists(LOG_PATH):
        unrecognized = load_json_file(LOG_PATH, [])
    else:
        unrecognized = []

    st.subheader("1. Unrecognized Query Logging Pool")
    st.caption(f"Total Low-Confidence Queries Logged for Admin Review: **{len(unrecognized)}**")

    if unrecognized:
        df_unrec = pd.DataFrame(unrecognized)
        st.dataframe(df_unrec, width="stretch")

        st.subheader("2. Admin Review Actions")

        configured_admin_pin = os.getenv("CHATBOT_ADMIN_PIN", "").strip()
        if not configured_admin_pin:
            st.warning("Read-only mode: set the CHATBOT_ADMIN_PIN environment variable before starting Streamlit to enable dataset changes.")
            admin_unlocked = False
        elif st.session_state.get("admin_unlocked", False):
            admin_unlocked = True
            st.success("Administrator actions are unlocked for this browser session.")
        else:
            supplied_pin = st.text_input("Administrator PIN", type="password")
            if st.button("Unlock administrator actions", width="stretch"):
                if supplied_pin == configured_admin_pin:
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect administrator PIN.")
            admin_unlocked = False

        col_q, col_tag = st.columns(2)
        with col_q:
            selected_query = st.selectbox("Select Query to Manage:", [item['query'] for item in unrecognized])
        with col_tag:
            selected_target_tag = st.selectbox("Assign Target Intent Tag (If merging):", [item['tag'] for item in data_json['intents']])

        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            if st.button("➕ Merge Query & Retrain Model", width="stretch", disabled=not admin_unlocked):
                normalized_query = " ".join(selected_query.split()).strip()
                existing_patterns = {
                    " ".join(pattern.split()).strip().casefold()
                    for intent in data_json["intents"]
                    for pattern in intent.get("patterns", [])
                }
                if not 3 <= len(normalized_query) <= 240:
                    st.error("The query must contain between 3 and 240 characters.")
                elif normalized_query.casefold() in existing_patterns:
                    st.error("This query already exists in the training dataset.")
                else:
                    for item in data_json["intents"]:
                        if item["tag"] == selected_target_tag:
                            item["patterns"].append(normalized_query)
                            break
                    remaining = [item for item in unrecognized if item.get("query") != selected_query]
                    atomic_write_json(DATASET_PATH, data_json)
                    atomic_write_json(LOG_PATH, remaining)
                    st.success(f"Merged the reviewed query into intent '{selected_target_tag}'.")
                    st.cache_resource.clear()
                    st.rerun()

        with btn_col2:
            if st.button("🗑️ Discard Selected Query", width="stretch", disabled=not admin_unlocked):
                unrecognized = [item for item in unrecognized if item['query'] != selected_query]
                atomic_write_json(LOG_PATH, unrecognized)
                st.warning("The selected logged query was discarded without modifying the training set.")
                st.rerun()

        with btn_col3:
            confirm_clear = st.checkbox("I confirm that all logged queries may be cleared.", disabled=not admin_unlocked)
            if st.button("🧹 Clear All Queries", width="stretch", disabled=not (admin_unlocked and confirm_clear)):
                atomic_write_json(LOG_PATH, [])
                st.info("All logged unrecognized queries have been cleared.")
                st.rerun()
    else:
        st.info("🎉 No unrecognized queries in the log pool yet! As students ask new questions with low confidence, they will automatically appear here for review.")

with tab_info:
    st.header("📄 Assignment Specifications & Requirements")
    st.markdown("""
    ### Project Overview
    - **Assignment Title**: Artificial Intelligence Assignment Specifications
    - **Topic Selected**: 5. Chatbot Development
    - **Team Size**: 2 Members
    - **Dataset**: Curated university FAQ intent dataset (`intents.json`; see the report and data source notes)

    ### Requirements Fulfilled (Requirement g.i & g.ii)
    - **Intent Metrics (g.i)**: Accuracy, Precision, Recall, Weighted F1-Score.
    - **Response Quality (g.ii)**: reported only when an independent reference-answer set is available; intent-template reuse is not counted as valid evidence.
    - **Dialogflow Integration**: embedded Messenger UI plus a reproducible Dialogflow ES export package.
    - **Multi-Model Support**: Member 1 (Dialogflow Platform) + Member 2 (Python ML Pipeline).
    - **Continuous Active Learning Loop**: Automatic logging and admin GUI for incremental dataset expansion or discarding.
    - **Sentiment & Urgency Analysis**: Automatic classification of student query urgency and emotional tone.
    - **Intent Dataset Analytics**: distribution charts for the curated training phrases; this is not presented as big data.
    """)
