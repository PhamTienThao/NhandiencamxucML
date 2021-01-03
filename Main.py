#Link dataset: https://www.kaggle.com/msambare/fer2013
#link tham khảo: https://viblo.asia/p/nhan-dien-cam-xuc-khuon-mat-don-gian-voi-keras-V3m5WvRwlO7#_dataset-1
# import những thư viện cần thiết
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#Khởi tạo mô hình: sử dụng những hàm quen thuộc trong Keras , ta xây dựng một models 
#Link để xem các hàm trong keras https://www.tensorflow.org/api_docs/python/tf/keras
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25)) # loại bỏ p% số lượng node trong layer đấy, hay nói cách khác là dữ lại (1-p%) node
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten()) #this converts our 3D feature maps to 1D feature 
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5') #load model đã được train
cv2.ocl.setUseOpenCL(False)
#Cảm xúc
emotion_dict = {0: "   Angry   ", 1: " Disgusted ", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ",
                5: "    Sad    ", 6: "Surprised"}
#biểu tượng cảm xúc
emoji_dist = {0: "./emojis/angry.png", 2: "./emojis/disgusted.png", 2: "./emojis/fearful.png", 3: "./emojis/happy.png",
              4: "./emojis/neutral.png", 5: "./emojis/sad.png", 6: "./emojis/surpriced.png"}
#sử dụng Camera của thiết bị
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() # Chụp hình ảnh từ camera
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load model phát hiện khuôn mặt: hàm trong opencv
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Chuyển từ ảnh màu sang ảnh xám bằng OpenCV
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5) # Phát hiện khuôn mặt trong khung hình
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # Gán nhãn cảm xúc dự đoán được lên hình
        roi_gray_frame = gray_frame[y:y + h, x:x + w] #lưu khuôn mặt đã chụp
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)  # Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48
        emotion_prediction = emotion_model.predict(cropped_img) #dự đoán cảm xúc đã được train
        maxindex = int(np.argmax(emotion_prediction)) # Lấy nhãn của cảm xúc có tỉ lệ cao nhất dự đoán được
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)   # In các mức độ của cảm xúc
    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC)) # Resize về kích thước 1200x860
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Nhấn phím "q" để kết thúc chương trình
        exit(0)
cap.release()
cv2.destroyAllWindows() # Dọn dẹp chương trình, giải phóng bộ nhớ và camera
