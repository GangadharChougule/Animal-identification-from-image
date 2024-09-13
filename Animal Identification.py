import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')


image = cv2.imread('image file path')
image = cv2.resize(image, (224,224))


image = np.expand_dims(image, axis=0)
image = preprocess_input(image)


preds = model.predict(image)
preds_decoded = decode_predictions(preds, top=1)[0][0]


print('This Is:', preds_decoded[1])
print('Probability:', preds_decoded[2])

if  preds_decoded[2] > 0.49:
  print("this is animal")
else:
  print("this not is Animal")

