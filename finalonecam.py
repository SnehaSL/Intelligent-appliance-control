import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import time
from keras.preprocessing import image
from threading import Thread

cutofftime = 60
frametime = 1
is_human1=False
result=False
result_time = time.time()
framestart_time = time.time()
# Load the pre-trained models
resnet50_model = keras.applications.ResNet50(weights='imagenet')

# Open the first camera
cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("Could not open camera 1")

# Set the resolution for both cameras
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def process_frames():
    global result,is_human1,is_human2
    image_path1 = 'temp1/Frame1.jpg'

    image1 = keras.preprocessing.image.load_img(image_path1, target_size=(224, 224))
    input_image1 = keras.preprocessing.image.img_to_array(image1)
    input_image1 = np.expand_dims(input_image1, axis=0)
    input_image1 = keras.applications.vgg16.preprocess_input(input_image1)

    resnet50_pred1 = resnet50_model.predict(input_image1)

    threshold = 0.5
    is_human1 = np.max(resnet50_pred1) > threshold
    result = result or is_human1

                
count=0
while True:
    # Read frames from both cameras
    ret1, frame1 = cap1.read()


    # Display the frames
    cv2.imshow('Camera 1', frame1)
 

    if time.time() - result_time <= cutofftime:
        if time.time() - framestart_time >= frametime:
    # Start the thread for processing frames
            cv2.imwrite('temp1/Frame1.jpg', frame1)
            frame_thread = Thread(target=process_frames)
            frame_thread.start()
            #frame_thread.join()
            framestart_time = time.time()
            if is_human1:
                count=count+1
                text = "Human Detected"
                print("frame1",text)
                cv2.putText(frame1, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)          
    else:
        if result:
            if count>=3:
                print("Fan is running")
            else:
                print("fun is turned off")

        else:
            print("Fan is turned off")
        result = False
        result_time = time.time()
        count=0
    
    #cv2.imshow('camera1', frame1)
    #cv2.imshow('camera2', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the video captures and close the windows
cap1.release()
cv2.destroyAllWindows()



