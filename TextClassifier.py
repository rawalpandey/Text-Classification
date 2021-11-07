
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print ('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello World'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for index in encoded_string:
  print ('{} ----> {}'.format(index, encoder.decode([index])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)

test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)
'''
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=30)

'''

model = keras.models.load_model('/home/rawal/Desktop/model.h5')
print(model.summary())
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)

# predict on a sample text without padding.

sample_pred_text1 = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
prediction1 = sample_predict(sample_pred_text1, pad=False)
#print (predictions)

# predict on a sample text with padding

sample_pred_text2 = ('The movie was not that good. The animation and the graphics '
                    'were not nice. I would not recommend this movie.')
prediction2 = sample_predict(sample_pred_text2, pad=True)
#print (predictions)


print("**************** Movie Review analysis ******************")
print("                                                          ")

print("Movie Review 1 : " ,sample_pred_text1 )
if(prediction1 >=0.5):
  print("-> Classification : This is a Positive movie review ")
else:

  print("-> Classification : This is a Negative movie review")
print("                                                          ")

print("Movie Review 2 : " ,sample_pred_text2 )
if(prediction2 >=0.5):
  print("-> Classification :  This is a Positive movie review ")
else:
  print("-> Classification :  This is a Negative movie review")