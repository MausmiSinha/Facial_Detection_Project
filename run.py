import cv2
import numpy as np
from keras.models import model_from_json

emotion_classes = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprise"}

#Loading Trained Model:
json_file = open(r"D:\Python\Projects\Facial_Detection_Project\model\model_v2.json", 'r')

loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#Loading Weights:
model.load_weights(r"D:\Python\Projects\Facial_Detection_Project\model\new_model_v2.h5")

print("Model Loaded Successfully")

#Starting the Webcam feed:
cap = cv2.VideoCapture(0)

while True:
    #Find Hear Cascade to draw box around face:
    ret, frame = cap.read()
    frame =  cv2.resize(frame, (1280,720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #Converting the Captured frame to gray scale:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces available on camera:
    num_face = face_detector.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5)

    # Take each fave available on the camera and preprocess it:
    for (x, y, w, h) in num_face:
        cv2.rectangle(frame, (x,y-50), (x+w, y+h+10), (0,255,0), 4)
        roi_gray_frame = gray_frame[y:y+h, x: x+w]
        cropped_img = np.expand_dims(cv2.resize(roi_gray_frame, (48,48), -1), 0)

        #Predict the emotion:
        if np.sum([roi_gray_frame])!=0:
            emotion_prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            label_position = (x,y)
            cv2.putText(frame,emotion_classes[maxindex],label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
