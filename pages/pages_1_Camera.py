import cv2
import logging
from collections import deque

import streamlit as st

from utils import SLInference

logger = logging.getLogger(__name__)

def main(config_path):
    """
    Main function of the app.
    """
    inference_thread = SLInference(config_path)
    inference_thread.start()

    # Set up OpenCV video capture
    cap = cv2.VideoCapture(0)

    gestures_deque = deque(maxlen=5)

    # Set up Streamlit interface
    st.title("Sign Language Recognition Demo")
    image_place = st.empty()
    text_output = st.empty()
    last_5_gestures = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame")
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_place.image(img_rgb)

        inference_thread.input_queue.append(cv2.resize(frame, (224, 224)))

        gesture = inference_thread.pred
        if gesture not in ['no', '']:
            if not gestures_deque:
                gestures_deque.append(gesture)
            elif gesture != gestures_deque[-1]:
                gestures_deque.append(gesture)

        text_output.markdown(f'<p style="font-size:20px"> Current gesture: {gesture}</p>',
                             unsafe_allow_html=True)
        last_5_gestures.markdown(f'<p style="font-size:20px"> Last 5 gestures: {" ".join(gestures_deque)}</p>',
                                 unsafe_allow_html=True)
        print(gestures_deque)

if __name__ == "__main__":
    main("configs/config.json")
