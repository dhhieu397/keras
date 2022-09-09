from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import cv2

#load dữ liệu:
(X_train, y_train), (X_test, y_test) =mnist.load_data()

plt.imshow(X_train[0])

X_train = X_train.reshape(60000, 28, 28,1)
X_test = X_test.reshape(10000, 28, 28, 1)

#one hot encode:
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()

#add more layers:
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict:
y_hat = model.predict(X_test[:4])
print(y_hat)

y_label =np.argmax(y_hat, axis=1)
print(y_label)
# print(X_train.shape, X_test.shape)
# cv2.imshow('image',X_train[0])
# print(X_train, y_train)
# print(X_test, y_test)
# print(X_train.shape)

# cv2.waitKey(0)
# 60k ảnh train, 10k ảnh test, mỗi ảnh kích thước 28 x 28
# vì là ảnh xám nên phải reshape chiều mỗi ảnh về (28,28,1)

#hàm softmax đầu vào là vector mà y_train hiện tại đang là số nguyên;
#to_categorical: chuyển thành one hot vector
#output của model: vector gồm 10 kí tự, từ 0-9.