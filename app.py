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
json_file = open(r"model/model_v2.json", 'r')

# Loading model.json file into model
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Loading Weights:
model.load_weights(r"model/new_model_v2.h5")

# Loading Face Cascade
try: 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Unable to load Cascade Classifier", icon="⚠️")

# Configuring Rtc
RTC_Configuration = RTCConfiguration({"iceServers": [{"url" : ["stun:stun.l.google.com:19302"]}]})

class EmotionDetector(VideoTransformerBase):
    def detect(self, frame):
        # Converting frame into 2 array of RGB format.
        img = frame.to_ndarray(format = "bgr24")

        #Converting the Captured frame to gray scale:
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces available on camera:
        num_face = face_detector.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5)

        # Take each fave available on the camera and preprocess it:
        for (x, y, w, h) in num_face:
            cv2.rectangle(img, (x,y-50), (x+w, y+h+10), (0,255,0), 4)
            roi_gray_frame = gray_frame[y:y+h, x: x+w]
            cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48,48), -1), 0)

            #Predict the emotion:
            if np.sum([roi_gray_frame])!=0:
                emotion_prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                label_position = (x,y)
                cv2.putText(frame,emotion_classes[maxindex],label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(img,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        return img


st.title("Hello Mausmi")
st.header('Hello Aman this site is made by mausmi')

