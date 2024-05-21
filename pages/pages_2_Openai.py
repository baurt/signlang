import logging
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import cv2
from model import Predictor  # Import Predictor from your model file
DEFAULT_WIDTH = 50

import openai

# Initialize the OpenAI client
openai.api_key  ='sk-proj-mWWgUXxgG6SWCbRj6QpkT3BlbkFJYhthNppsE5NA8IUxcBmd'

def correct_text_gpt3(input_text):
    prompt = f"Исправь грамматические ошибки в тексте: '{input_text}'"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects grammatical errors."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    corrected_text = response.choices[0].message['content'].strip()
    return corrected_text

#st.set_page_config(layout="wide")



width = 50
side = max((100 - width) / 1.2, 0.01)

_, container, _ = st.columns([side, width, side])

logger = logging.getLogger(__name__)

class SLInference:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.predictor = Predictor(self.config)
        self.input_queue = deque(maxlen=32)  # Queue to store 32 frames
        self.pred = ''

    def load_config(self, config_path):
        import json
        with open(config_path, 'r') as f:
            return json.load(f)

    def start(self):
        pass  # This method can be left empty or add initialization logic

    def predict(self, frames):
        frames_resized = [cv2.resize(frame, (224, 224)) for frame in frames]
        while len(frames_resized) < 32:
            frames_resized.append(frames_resized[-1])
        result = self.predictor.predict(frames_resized)
        if result:
            return result["labels"][0]
        return 'no'

def process_batch(inference_thread, frames, gestures):
    gesture = inference_thread.predict(frames)
    if gesture not in ['no', ''] and gesture not in gestures:
        gestures.append(gesture)

def main(config_path):
    #st.set_page_config(layout="wide")
    st.title("Sign Language Recognition Demo")

    st.warning("Please upload a video file for prediction.")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "gif"])

    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        container.video(data=video_bytes)
        #st.video(video_bytes)

        inference_thread = SLInference(config_path)
        inference_thread.start()

        text_output = st.empty()

        if st.button("Predict"):
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_bytes)
            cap = cv2.VideoCapture(tfile.name)

            gestures = []
            frames = []
            batch_size = 32

            def process_frames(batch):
                process_batch(inference_thread, batch, gestures)

            with ThreadPoolExecutor() as executor:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    if len(frames) == batch_size:
                        executor.submit(process_frames, frames)
                        frames = []

                if frames:
                    executor.submit(process_frames, frames)

            cap.release()
            text_output.markdown(f'<p style="font-size:20px"> Gestures in video: {" ".join(gestures)}</p>',
                                 unsafe_allow_html=True)
            st.text(correct_text_gpt3(" ".join(gestures)))
            
            print(gestures)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main("configs/config.json")
