# --coding:utf-8--
import sys
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input

target_size = (224, 224)

def predict(model, img, target_size):
    """Run model prediction on image
    Args:
      model: keras model
      img: PIL format image
      target_size: (w,h) tuple
    Returns:
      list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


# plot
labels = ("0","1","2","3","4","5","6","7","8","9")

def plot_preds(image, preds, labels):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
      image: PIL image
      preds: list of predicted labels and their probabilities
    """
    # plt.imshow(image)
    # plt.axis('off')
    # plt.figure()
    # plt.barh([0, 1], preds, alpha=0.5)
    # plt.yticks([0, 1], labels)
    # plt.xlabel('Probability')
    # plt.xlim(0, 1.01)
    # plt.tight_layout()
    # plt.show()


# import model
model = load_model('./model/checkpoint-19e-val_accuracy_0.99.hdf5')

# Local picture prediction
img = Image.open('0.jpg')

preds = predict(model, img, target_size)
# plot_preds(img, preds, labels)
print(type(labels))
preds = preds.tolist()
print(preds)
p=preds.index(max(preds))

print(labels[p])
