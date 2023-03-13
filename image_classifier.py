import tensorflow as tf
import numpy as np

# загрузка набора данных
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# нормализация данных
train_images = train_images / 255.0
test_images = test_images / 255.0

# определение модели
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

# компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# обучение модели
model.fit(train_images, train_labels, epochs=10, 
          validation_data=(test_images, test_labels))

# сохранение модели
model.save('image_classifier.h5')
