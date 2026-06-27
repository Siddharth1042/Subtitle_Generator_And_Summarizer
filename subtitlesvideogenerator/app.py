import streamlit as st
import whisper
from transformers import pipeline
import imageio_ffmpeg
import subprocess
import os

st.title("AI Video → Transcript + Summary Generator")

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")


# ---------------------------
# Upload video
# ---------------------------
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:

    video_path = uploaded_file.name

    # save file
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

    # ---------------------------
    # Generate button
    # ---------------------------
    if st.button("Generate Transcript & Summary"):

        st.info("Extracting audio...")

        audio_path = "audio.wav"

        # ---------------------------
        # FAST FFmpeg (NO INSTALL REQUIRED)
        # ---------------------------
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run([
            ffmpeg_path,
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            audio_path,
            "-y"
        ])

        st.success("Audio extracted!")

        # ---------------------------
        # Whisper transcription
        # ---------------------------
        st.info("Loading Whisper model...")
        model = load_whisper()

        st.info("Transcribing...")
        result = model.transcribe(audio_path)
        transcript = result["text"]

        st.subheader("Transcript")
        st.write(transcript)

        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        # ---------------------------
        # Summarization
        # ---------------------------
        st.info("Generating summary...")
        summarizer = load_summarizer()

        # safety limit
        short_text = transcript[:2000]

        summary = summarizer(
            short_text,
            max_length=150,
            min_length=50,
            do_sample=False
        )

        final_summary = summary[0]["summary_text"]

        st.subheader("Summary")
        st.write(final_summary)

        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)

        # ---------------------------
        # Download buttons
        # ---------------------------
        st.download_button(
            "Download Transcript",
            transcript,
            file_name="transcript.txt"
        )

        st.download_button(
            "Download Summary",
            final_summary,
            file_name="summary.txt"
        )