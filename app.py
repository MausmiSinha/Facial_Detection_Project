# Importing Necessary Libraries
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase,WebRtcMode

# Setting Page Title:
st.set_page_config(page_title="Emotion Detector", initial_sidebar_state = 'auto')

# Initializing WebRTC Streamer
webrtc_streamer(key="sample")

# Declaring Classes
emotion_classes = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprise"}

# Loading Trained Model:
json_file = open(r"D:\Python\Projects\Facial_Detection_Project\model\model_v2.json", 'r')

# Loading model.json file into model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading Weights:
model.load_weights(r"D:\Python\Projects\Facial_Detection_Project\model\new_model_v2.h5")

# Loading Face Cascade
try: 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Unable to load Cascade Classifier", icon="⚠️")

# Configuring Rtc
RTC_Configuration = RTCConfiguration({"iceServers": [{"url" : ["stun:stun.l.google.com:19302"]}]})



st.title("Hello Mausmi")
st.header('Hello Aman this site is made by mausmi')

