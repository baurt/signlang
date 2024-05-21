import logging
import queue
from collections import deque

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils import SLInference


logger = logging.getLogger(__name__)

def main():
    """
    Main function of the app.
    """
    config = {
        "path_to_model": "S3D.onnx",
        "threshold": 0.8,
        "topk": 5,
        "path_to_class_list": "RSL_class_list.txt",
        "window_size": 8,
        "provider": "OpenVINOExecutionProvider"
    }

    inference_thread = SLInference(config)
    inference_thread.start()

    webrtc_ctx = webrtc_streamer(
        key="video-sendonly",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": True},
    )

    gestures_deque = deque(maxlen=5)

    # Set up Streamlit interface
    st.title("Sign Language Recognition Demo")
    image_place = st.empty()
    text_output = st.empty()
    last_5_gestures = st.empty()


    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty")
                continue

            img_rgb = video_frame.to_ndarray(format="rgb24")
            image_place.image(img_rgb)
            inference_thread.input_queue.append(video_frame.reformat(224,224).to_ndarray(format="rgb24"))

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
    main()