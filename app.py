# Importing Necessary Libraries
import cv2
import numpy as np
from keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, VideoTransformerBase

# Setting Page Title:
st.set_page_config(page_title="Smile", initial_sidebar_state = 'auto')

css_example = '''                                           
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">    
    
    <style>
        .bodyP1{
            font-size: 20px;
        }
        .footer{
            display: flex;
            justify-content:center;
            align-items: center;
            font-size: 20px;
            font-weight: 300;
            margin-top: 50px;
        }
        .aboutUs p{
            font-size: 18px;
            text-align: justify;
        }
    </style>
'''
st.write(css_example, unsafe_allow_html=True)

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
    activiteis = ["Home", "About Us"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)    
    html_sidebar =  """
        <div align="center" style="text-align:center">
            <h2>Developed By: Aman & Mausmi</h2>
            <strong>Email:</strong> <a target="_blank" href="mailto:bhogalamansingh22@gmail.com"> bhogalamansingh22@gmail.com</a> & <a target="_blank" href="mailto:sinhatanu2001@gmail.com">sinhatanu2001@gmail.com</a>
            <br>
            <a target="_blank" href="https://github.com/MausmiSinha/Facial_Detection_Project" style="font-size:36px"> <i class="fa-brands fa-github"></i> </a>
        </div> 
    """
    st.sidebar.markdown(html_sidebar, unsafe_allow_html=True)

    # Menu Cases:
    if choice == "Home":
        st.title("Smile")
        html_temp_home1 = """
                            <div>
                                <p class="bodyP1">Welcome to Smile. This is an emotion recognition web application! Our advanced technology combines OpenCV and Convolutional Neural Networks to accurately identify emotions in real-time. </p>
                                <p class="bodyP1">Our application is user-friendly and easy to use. Simply use your computer's camera to capture a live video feed, and our advanced algorithms will quickly analyze the facial expressions to detect the emotion displayed. Our application can recognize a wide range of emotions, including happiness, sadness, anger, surprise, fear, and disgust.</p>
                                <h2>Try Now</h2>
                                <p>Click on start to use webcam and detect your face emotion</p>
                            </div>
                            """
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)
        html_home2  = """
            <div class="footer">
                <p>&copy; 2023 by Aman Singh Bhogal & Mausmi Sinha. All rights reserved.</p>
            </div>
        """
        st.write(html_home2, unsafe_allow_html=True)
    elif choice == "About Us":
        st.title("About Us:")
        html_temp_about1= """
                            <div class="aboutUs">
                                <p>
                                    This Emotion Detection Web Application is created by Aman Singh Bhogal and Mausmi Sinha as their Machine Learning Subject's Project. The application uses OpenCV and Convolutional Neural Network model to accurately detect emotions in real-time.
                                </p>
                                <p>
                                    Together, Aman and Mausmi worked on this project to create an easy-to-use web application that can accurately detect a wide range of emotions. They hope that this application can be useful for researchers, educators, and anyone interested in exploring the field of emotion detection.
                                </p>
                                <p>
                                    If you have any questions or feedback about this project, please feel free to contact us at <a target="_blank" href="mailto:bhogalamansingh22@gmail.com"> bhogalamansingh22@gmail.com</a> or <a target="_blank" href="mailto:sinhatanu2001@gmail.com">sinhatanu2001@gmail.com</a>
                                </p>
                            </div>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)
        st.title("Authors:")
        st.image(['images/Aman Singh Bhogal.jpg','images/mausmi.jpg'], caption=["Aman Singh Bhogal", "Mausmi Sinha"], width=300)
    else:
        pass


if __name__ == "__main__":
    main()


