import streamlit as st
import tempfile
import os
from speech_to_text import speech_to_text, filler_word_ratio
from semantic_eval import semantic_similarity, auto_detect_topic, get_gemini_feedback, get_reference, CONCEPT_REFERENCES
from audio_utils import extract_audio_features, save_waveform
from scoring_engine import evaluate_understanding, get_score_emoji, calculate_fluency_score
from report_generator import generate_pdf_report

# Page config
st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: white; }
    .stApp { background-color: #0d1117; }
    h1, h2, h3 { color: white; }
    .metric-card {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        text-align: center;
    }
    .score-card {
        background-color: #16213e;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>🎙️ Voice-Based Concept Understanding Analyser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Automated evaluation of spoken conceptual explanations using AI</p>", unsafe_allow_html=True)
st.markdown("---")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    gemini_api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="Enter your Gemini API key")
    st.markdown("---")
    auto_detect = st.checkbox("🤖 Auto Detect Topic (Gemini)", value=False)
    if not auto_detect:
        topic = st.selectbox("📚 Select Concept Topic", list(CONCEPT_REFERENCES.keys()))
    language = st.radio("🌐 Language", ["English", "Telugu (Beta ⚠️)"])
    lang_code = "te" if "Telugu" in language else "en"
    if "Telugu" in language:
        st.warning("Telugu transcription may have minor errors!")
    st.markdown("---")
    st.markdown("### 📊 Attempt History")
    if st.session_state.history:
        for i, h in enumerate(st.session_state.history[-5:]):
            st.markdown(f"**Attempt {i+1}:** {h['topic']} → {h['score']}/100")
    else:
        st.info("No attempts yet!")

# Main layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 📤 Upload Student Audio (WAV/MP3)")
    audio_file = st.file_uploader("Drag and drop file here", type=["wav", "mp3"], label_visibility="collapsed")
    if audio_file:
        st.audio(audio_file)

with col2:
    if not auto_detect:
        st.markdown("### 📖 Concept Reference")
        ref_text = get_reference(topic)
        st.markdown(f"""<div style='background:#1a1a2e; padding:15px; border-radius:10px; color:white; font-size:14px;'>{ref_text}</div>""", unsafe_allow_html=True)

st.markdown("---")

# Analyze button
if audio_file:
    if st.button("🔍 Analyze Concept Understanding"):
        with st.spinner("Processing and evaluating..."):
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

            # Run all modules
            transcript_result = speech_to_text(audio_path, lang_code)
            audio_features = extract_audio_features(audio_path)
            waveform_bytes = save_waveform(audio_path)

            if not transcript_result["success"]:
                st.error(f"Transcription failed: {transcript_result.get('error')}")
            else:
                transcript = transcript_result["text"]
                filler = filler_word_ratio(transcript)

                # Auto detect topic
                if auto_detect and gemini_api_key:
                    topic = auto_detect_topic(transcript, gemini_api_key)
                    st.info(f"🤖 Auto detected topic: **{topic}**")

                ref_text = get_reference(topic)
                similarity = semantic_similarity(transcript, ref_text)
                score, level, color = evaluate_understanding(similarity, filler, audio_features)
                fluency = calculate_fluency_score(filler, audio_features.get("pause_ratio", 0))
                emoji = get_score_emoji(level)

                # Gemini feedback
                feedback = ""
                if gemini_api_key:
                    feedback = get_gemini_feedback(transcript, topic, score, gemini_api_key)
                else:
                    feedback = "Add Gemini API key in sidebar for personalized AI feedback!"

                # Save to history
                st.session_state.history.append({
                    "topic": topic,
                    "score": score,
                    "level": level
                })

                # Store result
                st.session_state.result = {
                    "transcript": transcript,
                    "topic": topic,
                    "ref_text": ref_text,
                    "similarity": similarity,
                    "filler": filler,
                    "audio_features": audio_features,
                    "score": score,
                    "level": level,
                    "color": color,
                    "emoji": emoji,
                    "fluency": fluency,
                    "feedback": feedback,
                    "waveform_bytes": waveform_bytes,
                    "audio_path": audio_path
                }

            os.unlink(audio_path)

# Show results
if st.session_state.result:
    r = st.session_state.result
    st.markdown("---")
    st.markdown("<div style='background:#1a1a2e; padding:10px; border-radius:8px;'><h3 style='color:#00b4d8;'>✅ Analysis Completed</h3></div>", unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📝 Transcribed Explanation")
        st.markdown(f"""<div style='background:#16213e; padding:15px; border-radius:10px; color:white;'>{r['transcript']}</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🏆 Final Evaluation")
        st.markdown(f"""
        <div class='score-card'>
            <h2 style='color:white;'>Understanding Score</h2>
            <h1 style='color:{r['color']}; font-size:60px;'>{r['score']}/100</h1>
            <h2 style='color:{r['color']};'>{r['emoji']} {r['level']}</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'><h4 style='color:gray;'>Semantic Similarity</h4><h2 style='color:#00b4d8;'>{r['similarity']}</h2></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'><h4 style='color:gray;'>Filler Word Ratio</h4><h2 style='color:#00b4d8;'>{r['filler']}</h2></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'><h4 style='color:gray;'>Confidence (Energy)</h4><h2 style='color:#00b4d8;'>{r['audio_features'].get('rms_energy', 0)}</h2></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Waveform
    if r['waveform_bytes']:
        st.markdown("### 🔊 Audio Visualization")
        st.image(r['waveform_bytes'], use_column_width=True)

    # Evaluation summary table
    st.markdown("### 📊 Evaluation Summary")
    st.table({
        "Metric": ["Semantic Similarity", "Filler Word Ratio", "Pause Ratio", "Confidence (Energy)", "Fluency Score", "Final Score", "Understanding Level"],
        "Value": [
            r['similarity'],
            r['filler'],
            r['audio_features'].get('pause_ratio', 0),
            r['audio_features'].get('rms_energy', 0),
            f"{r['fluency']['fluency_score']}/100",
            f"{r['score']}/100",
            r['level']
        ]
    })

    # Gemini feedback
    if r['feedback']:
        st.markdown("### 🤖 AI Feedback")
        st.markdown(f"""<div style='background:#1a1a2e; padding:15px; border-radius:10px; color:white;'>{r['feedback']}</div>""", unsafe_allow_html=True)

    # PDF Download
    st.markdown("---")
    pdf_bytes = generate_pdf_report(
        topic=r['topic'],
        reference_text=r['ref_text'],
        transcribed_text=r['transcript'],
        similarity=r['similarity'],
        filler_ratio=r['filler'],
        pause_ratio=r['audio_features'].get('pause_ratio', 0),
        rms_energy=r['audio_features'].get('rms_energy', 0),
        final_score=r['score'],
        understanding_level=r['level'],
        fluency_score=r['fluency']['fluency_score'],
        gemini_feedback=r['feedback'],
        waveform_bytes=r['waveform_bytes']
    )
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"VBCUA_Report_{r['topic'].replace(' ', '_')}.pdf",
        mime="application/pdf"
    )
else:
    st.markdown("<p style='text-align:center; color:gray;'>Upload an audio file to begin analysis.</p>", unsafe_allow_html=True)
