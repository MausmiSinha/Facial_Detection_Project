# Importing Necessary Libraries
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, VideoTransformerBase

# Setting Page Title:
st.set_page_config(page_title="Emotion Detector", initial_sidebar_state = 'auto')

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

print("Model lodded scussesfully")

# Loading Face Cascade
try: 
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.error("Unable to load Cascade Classifier", icon="⚠️")


class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        # Converting frame into 2 array of RGB format.
        img = np.array(frame.to_ndarray(format = "bgr24"))

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
                output = str(emotion_classes[maxindex])
                cv2.putText(img,output,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(img,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        return img


def main():
    # Face Analysis Application #
    st.title("Emotion Detector")
    activiteis = ["Home", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Aman and Mausmi    
            Email : bhogalamansingh22@gmail.com and sinhatanu2001@gmail.com
            [LinkedIn] (https://www.linkedin.com/in/aman-singh-bhogal-002914195/)
            [LinkedIn] (https://www.linkedin.com/in/mausmi-sinha-5a3345220/)
            """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Emotion detector using OpenCV, Custom CNN model and Streamlit.</h4>
                                            <p>This project aims to build a Facial Emotion Recognition system using the OpenCV library and Convolutional Neural Network (CNN) algorithms. The system will be trained on a dataset of images of faces with annotations of emotions to predict the emotions of new unseen images.</p>
                                            </div>
                                            </br>
                            <div>
                                <h2>Try Now</h2>
                                <p>Click on start to use webcam and detect your face emotion</p>
                            </div>
                            """
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
    elif choice == "About":
        st.subheader("About us")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()


