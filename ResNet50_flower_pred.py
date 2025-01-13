import numpy as np
import cv2 
import keras 
from keras import layers
from keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
data_dir = pathlib.Path('datasets/flower_photos')
roses = list(data_dir.glob('roses/*'))


height,width = 224,224
batch_size = 32

trainds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed =123,
    label_mode='categorical',
    image_size=(height,width),
    batch_size=batch_size
)

validationds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed =123,
    label_mode='categorical',
    image_size=(height,width),
    batch_size=batch_size
)


classNames = ['daisy','dandelion','roses','sunflowers','tulips']

model = Sequential()

pretrained_model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling='max',
    classes=1000  ,
    classifier_activation="softmax",
)

for layer in pretrained_model.layers:
    layer.trainable= False

model.add(pretrained_model)
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='sigmoid'))
model.add(layers.Dense(5,activation='softmax'))

model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(trainds,validation_data = validationds,epochs=10)


image1 = cv2.imread(roses[0])
image1 = cv2.resize(image1,(224,224))
image1 = np.expand_dims(image1,axis=0)

pred = model.predict(image1)
print(pred)
output = classNames[np.argmax(pred)]
print(output)


# imageshape = (224,224)

# model = Sequential()
# model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenetv2/classification/4",input_shape = imageshape+(3,)))
# model.add(layers.Dense(10,activation="sigmoid"))

# bird = image.open("jay.jpg").resize(imageshape)
# bird = np.array(bird)/255
# result = model.predict(bird[np.newaxis,...])

# prediction = np.argmax(result)
# print(prediction)