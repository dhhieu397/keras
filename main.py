#1.Load dữ liệu và  chia Train, val, test:
from keras import Sequential
from keras.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split

dataset =loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# print(dataset)
X  =dataset[:,0:8]
y= dataset[:,8]

X_train_val, X_test, y_train_val, y_test  = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val =train_test_split(X_train_val, y_train_val, test_size=0.2)


model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# complie
# Phân loại 2 lớp: dùng loss: binary_cross_entropy
# Phân loại nhiều lớp: dùng loss: categorical cross entropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train model: epoch, batch_size, validation:
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val))

#save model:
model.save("mymodel.h5")


