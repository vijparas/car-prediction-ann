import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import  save_model

dataset = pd.read_csv('car.data.csv')
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values
#df = pd.DataFrame(['a','b','c','d','a','c','a','d'], columns=X[:,0])
labelencoder_X_Buying = LabelEncoder()
X[:,0]=labelencoder_X_Buying.fit_transform(X[:,0])
labelencoder_X_Maint = LabelEncoder()
X[:,1]=labelencoder_X_Maint.fit_transform(X[:,1])
labelencoder_X_doors = LabelEncoder()
X[:,2]=labelencoder_X_Maint.fit_transform(X[:,2])
labelencoder_X_persons = LabelEncoder()
X[:,3]=labelencoder_X_persons.fit_transform(X[:,3])
labelencoder_X_lug = LabelEncoder()
X[:,4]=labelencoder_X_lug.fit_transform(X[:,4])
labelencoder_X_safety = LabelEncoder()
X[:,5]=labelencoder_X_safety.fit_transform(X[:,5])

labelencoder_y=LabelEncoder()
y[:]=labelencoder_y.fit_transform(y[:])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y.reshape(len(y), 1)
categories = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=0)

classifier = Sequential()

classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu', input_dim=6))

classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=4, kernel_initializer='uniform', activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
np.argmax(y_pred,axis=1)

probability=classifier.predict_proba(X_test)
classes=classifier.predict_classes(X_test)
cm = confusion_matrix(np.argmax(y_test,axis=1),classes)