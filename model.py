import cv2,os
from keras.models import load_model
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv1D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import BatchNormalization

#
data_path='dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))
print(label_dict)
print(categories)
print(labels)
#
data = []
target = []
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Coverting the image into gray scale
            resized = cv2.resize(gray, (50,200))
            # resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[category])
            # appending the image and the label(categorized) into the list (dataset)
        except Exception as e:
            print('Exception:', e)
            # if any exception rasied, the exception will be printed here. And pass to the next image
#
import numpy as np
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],50,200,1))
target=np.array(target)
from keras.utils import np_utils
new_target=np_utils.to_categorical(target)
np.save('data',data)
np.save('target',new_target)
import numpy as np
data=np.load('data.npy')
target=np.load('target.npy')
# adding neural network layers
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers
model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(32,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#testing using train test split
from sklearn.model_selection import train_test_split
train_data,test_data,train_target,test_target=train_test_split(data,target,
                                                               test_size=1)
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',
                             monitor='val_loss',verbose=0,
                             save_best_only=True,
                             mode='auto')
history=model.fit(train_data,train_target,
                  epochs=20,
                  callbacks=[checkpoint],
                  validation_split=0)
print(model.evaluate(test_data,test_target))
model.save("mask training.hdf5")



