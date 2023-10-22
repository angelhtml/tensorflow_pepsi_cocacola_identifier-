import PIL.Image
import tensorflow as tf

val_ds = tf.keras.utils.image_dataset_from_directory(
  "./input/test",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(25, 25),
  batch_size=32)

train_ds = tf.keras.utils.image_dataset_from_directory(
  "./input/train",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(25, 25),
  batch_size=32)

num_classes = 2

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=6
)

model.save('saved_model/my_model')