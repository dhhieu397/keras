from keras.models import load_model
import numpy

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

model = load_model("mymodel.h5")


#evaluation trên tập test xem loss là bn:
loss, acc = model.evaluate(X_test, y_test)
print("Loss: ",loss)
print("Acc:", acc)

X_new = X_test[10]
y_new = y_test[10]
#convert giá trị  x_new thành tensor chỉ cần thêm chiều là đc :
X_new =numpy.expand_dims(X_new, axis=0)


y_predict = model.predict(X_new)
result ="Tiểu đường (1)"
if y_predict <=0.5:
    result ="Không tiểu đường (0)"

print(" gia tri dụ doán y: ", result)
print("Gia tri du doan y_predict =", y_predict)
print("Gia tri du doan dung y label =", y_new)
