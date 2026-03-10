from turtle import color
import cv2
import numpy as np
import streamlit as st
import requests
from datetime import timedelta
from typing import Optional

BACKEND_URL = "http://localhost:8000"
SUPPORTED_FORMATS = ["mp4", "avi", "mov"]
MAX_FILE_SIZE_MB = 200
VIDEO_PROCESSING_TIMEOUT = 300

def init_session_state():
    defaults = {
        "processed_video": None,
        "search_results": None,
        "upload_status": None,
        "backend_online": False
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=3)
        st.session_state.backend_online = response.status_code == 200
    except requests.RequestException:
        st.session_state.backend_online = False

def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    parts = str(td).split(":")
    return f"{parts[1]}:{parts[2][:2]}" if td < timedelta(hours=1) else f"{parts[0]}:{parts[1]}:{parts[2][:2]}"

def handle_search(query: str) -> Optional[dict]:
    if not st.session_state.backend_online:
        st.error("Backend API is unreachable. Please check the server.")
        return None

    try:
        with st.spinner("🔍 Searching video content..."):
            response = requests.post(
                f"{BACKEND_URL}/search",
                json={"prompt": query},
                timeout=15
            )
            response.raise_for_status()
            return response.json()
    except requests.RequestException as e:
        st.error(f"Search failed: {str(e)}")
        return None

def display_match(match: dict, index: int):
    confidence = match['score']

    CONFIDENCE_LEVELS = [
        (0.8, "✅ Very confident", "#4CAF50"),
        (0.6, "✔️ Confident", "#2196F3"),
        (0.4, "⚠️ Possible", "#FF9800"),
        (0.0, "❌ Unlikely", "#F44336")
    ]

    title_text = next((text for threshold, text, _ in CONFIDENCE_LEVELS if confidence > threshold), "❌ Unlikely")

    plain_text = title_text.replace("✅ ", "").replace("✔️ ", "").replace("⚠️ ", "").replace("❌ ", "")

    with st.expander(f"Match {index} | {title_text}", expanded=confidence > 0.6):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Timestamp", format_timestamp(match["timestamp"]))
        with col2:
            st.progress(float(confidence), text=f"{plain_text}: {confidence:.1%}")

        st.markdown(
            f"""<div style="height: 4px; background: linear-gradient(90deg, {color} 0%, {color} {confidence*100}%, #e0e0e0 {confidence*100}%, #e0e0e0 100%); border-radius: 2px; margin-top: 8px"></div>""",
            unsafe_allow_html=True
        )

        if confidence > 0.35:
            display_frame_preview(match["timestamp"], match["id"])

def display_frame_preview(timestamp: float, frame_id: str):
    try:
        response = requests.get(f"{BACKEND_URL}/frame/{frame_id}", timeout=5)
        response.raise_for_status()

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(
            img_rgb,
            caption=f"Timestamp: {format_timestamp(timestamp)}",
            use_column_width=True
        )
    except Exception as e:
        st.warning(f"Couldn't load frame: {str(e)}")

def handle_upload(video_file):
    if not st.session_state.backend_online:
        st.error("Backend API is unreachable. Uploads disabled.")
        return

    try:
        with st.status("📤 Uploading video...", expanded=True) as status:
            st.write("Validating file...")
            file_size_mb = video_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

            st.write("Processing frames...")
            files = {"file": (video_file.name, video_file.getvalue(), video_file.type)}
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                timeout=VIDEO_PROCESSING_TIMEOUT
            )
            response.raise_for_status()

            result = response.json()
            st.session_state.processed_video = video_file.name
            st.session_state.upload_status = "success"
            st.success(f"Processed {result['frames_processed']} frames")

    except Exception as e:
        st.session_state.upload_status = "error"
        st.error(f"Upload failed: {str(e)}")

def render_upload_section():
    with st.container(border=True):
        st.markdown("#### 📤 Upload Video")
        st.caption("Supported formats: MP4, AVI, MOV (max 200MB)")

        video_file = st.file_uploader(
            "Choose a video file",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed"
        )

        if video_file:
            if st.button("Process Video", type="primary", use_container_width=True):
                handle_upload(video_file)

def render_search_section():
    with st.container(border=True):
        st.markdown("#### 🔍 Search Video Content")

        if not st.session_state.processed_video:
            st.info("Upload a video first")
            return

        query = st.text_input(
            "Describe what you're looking for...",
            placeholder="e.g., 'a person wearing red', 'a dog playing with a ball'",
            label_visibility="collapsed"
        )

        if st.button("Search", type="primary", use_container_width=True):
            if not query:
                st.warning("Enter a search query")
            else:
                st.session_state.search_results = handle_search(query)

        if st.session_state.search_results:
            st.divider()
            if not st.session_state.search_results.get("results"):
                st.info("No matches found")
            else:
                st.caption(f"Found {len(st.session_state.search_results['results'])} matches")
                for i, match in enumerate(st.session_state.search_results["results"], 1):
                    display_match(match, i)

def render_sidebar():
    with st.sidebar:
        st.markdown("### Technical Details")
        st.markdown("""
        - **Backend**: FastAPI (Python)
        - **Search**: CLIP embeddings + ChromaDB
        - **Frame Processing**: OpenCV
        """)
        st.divider()
        status_color = "green" if st.session_state.backend_online else "red"
        st.markdown(f"Backend Status: :{status_color}[{'✅ Online' if st.session_state.backend_online else '❌ Offline'}]")

def main():
    st.set_page_config(
        page_title="Video Search Engine",
        page_icon="🎥",
        layout="centered"
    )
    init_session_state()
    check_backend_health()

    st.title("Video Content Search Engine")
    st.caption("Backend-powered semantic search for videos")
    st.divider()

    render_upload_section()
    st.divider()
    render_search_section()
    render_sidebar()

if __name__ == "__main__":
    main()
