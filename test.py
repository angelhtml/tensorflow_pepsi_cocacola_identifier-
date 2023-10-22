import numpy as np
import tensorflow as tf
from keras.models import load_model
from colored import fg, bg, attr

train_ds = tf.keras.utils.image_dataset_from_directory(
  "./input/train",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(25, 25),
  batch_size=32)

filepath = 'saved_model/my_model'
model = load_model(filepath, compile = True)

image = tf.keras.utils.load_img("input/test/pepsi/18.jpg",target_size=(25,25))
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)

print(train_ds.class_names)
print(predictions)

if predictions[0][0] > .5:
    print(f'{fg(196)}{"cocacola"}{attr(0)}')
elif predictions[0][1] > .5:
    print(f'{fg(12)}{"pepsi"}{attr(0)}')
else:
    print("not sure!")