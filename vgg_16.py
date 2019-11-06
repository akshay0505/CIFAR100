import pandas as pd
import numpy as np
from keras.models import Model , Sequential
from keras.layers import Dense, Activation , Conv2D, Conv3D,Flatten , MaxPool2D, MaxPooling2D, Dropout , Add, Input, AveragePooling2D, UpSampling2D , ZeroPadding2D
import os
import sys
from keras.applications.resnet50 import preprocess_input, decode_predictions , ResNet50
from keras.utils import to_categorical
from keras.optimizers import rmsprop , adam, Adagrad , SGD
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import BatchNormalization
from keras.initializers import glorot_uniform
from keras import regularizers

seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)


data_dir = "~/COL341/A3/Data/"
model_name = "model_vgg_16.json"
weight_name = "weight_vgg_16.h5"
pred_name = "pred3.txt"

num_classes = 100
weight_decay = 0.0005

train_data = pd.read_csv(sys.argv[1],header=None,delimiter=" ")
test_data  = pd.read_csv(sys.argv[2],header=None,delimiter=" ")
print(train_data.shape,test_data.shape)


train_image = train_data.iloc[:,:-2].values
R = train_image[:,0:1024].reshape(train_image.shape[0],32,32)
G = train_image[:,1024:2048].reshape(train_image.shape[0],32,32)
B = train_image[:,2048:3072].reshape(train_image.shape[0],32,32)
images_train = np.stack((R.T,G.T,B.T),axis=0).T

test_image = test_data.iloc[:,:-2].values
R = test_image[:,0:1024].reshape(test_image.shape[0],32,32)
G = test_image[:,1024:2048].reshape(test_image.shape[0],32,32)
B = test_image[:,2048:3072].reshape(test_image.shape[0],32,32)
image_test = np.stack((R.T,G.T,B.T),axis=0).T

def normalise(X):
    mean = np.mean(X,axis=(0,1,2,3))
    std = np.std(X, axis=(0, 1, 2, 3))
    X = (X-mean)/(std+1e-7)
    return X

x_train , x_val = normalise(images_train[:40000,:]) , normalise(images_train[40000:,:])
y_train , y_val = train_data.iloc[:40000,-1].values , train_data.iloc[40000:,-1].values
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

x_test = normalise(image_test)
print(x_train.shape,x_val.shape,y_train.shape,y_val.shape,x_test.shape)

def vgg_16(x_train):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',input_shape=x_train[0].shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
def vgg(X_train):
    inputs = Input(shape=X_train[0].shape)
    x = Conv2D(filters=32, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(inputs)
    x = Conv2D(filters=32, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=64, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = Conv2D(filters=64, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = Conv2D(filters=128, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=512, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = Conv2D(filters=512, kernel_size=(3,3),kernel_initializer=glorot_uniform(seed=0),activation="relu",padding="same")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512,activation="relu",kernel_initializer=glorot_uniform(seed=0))(x)
    x = Dense(256,activation="relu",kernel_initializer=glorot_uniform(seed=0))(x)
    x = Dense(100,activation="softmax")(x)
    model = Model(inputs=inputs,output=x)
    return model
# model = vgg(x_train)


model = vgg_16(x_train)
model.summary()



datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False, 
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,  
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
) 
datagen.fit(x_train)

def lr_scheduler(epoch):
    return learning_rate*(0.5**(epoch//lr_drop))
reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)
filepath = weight_name
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 128
maxepoches = 200
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
hist = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),steps_per_epoch=x_train.shape[0]//batch_size,epochs=maxepoches,validation_data=(x_val, y_val),callbacks=[reduce_lr,checkpoint],verbose=1)


# y_pred  = model.predict(x_test)
# y_class = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
# y = [i for i in range(0,100)]
# y = np.array(y).reshape(100,1)
# np.savetxt(pred_name,np.matmul(y_class,y).flatten())

model_json = model.to_json()
with open(model_name, "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")

from keras.models import model_from_json

json_file = open(model_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
new_model = model_from_json(loaded_model_json)
new_model.load_weights(weight_name)
print("Loaded model from disk")
 
y_pred  = new_model.predict(x_test)
y_class = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
y = [i for i in range(0,100)]
y = np.array(y).reshape(100,1)
np.savetxt(sys.argv[3],np.matmul(y_class,y).flatten())


