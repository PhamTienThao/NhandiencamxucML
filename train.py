import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


train_dir = 'train'
val_dir = 'test'
train_datagen = ImageDataGenerator(rescale=1./255) #feature nằm trong khoảng [0,1]
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48), #Kích thước sample 48x48
        batch_size=64, #Training Batch Size
        color_mode="grayscale", #màu gray của ảnh
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64, #số sample dùng cho 1 lần train
        color_mode="grayscale",
        class_mode='categorical')
#print(validation_generator)
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25)) # loại bỏ p% số lượng node trong layer đấy, hay nói cách khác là dữ lại (1-p%) node
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten()) # #this converts our 3D feature maps to 1D feature 
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax')) #Softmax chuyển đổi một vectơ thực thành một vectơ xác suất phân loại.

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64, # Mặc định sẽ là số lượng dữ liệu chia cho batch_size. Hiểu đơn giản là mỗi epoch sẽ dùng hết các dữ liệu để tính 
        epochs=50, #số lượng epoch thực hiện trong quá trình traning.
        validation_data=validation_generator,
        validation_steps=7178 // 64)
emotion_model.save_weights('model.h5')