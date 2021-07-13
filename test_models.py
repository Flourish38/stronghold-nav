import tensorflow as tf
import numpy as np

model = tf.saved_model.load("models/rl_rnn_1/model")

input = tf.random.normal((1, 1, 115))

model(input)

lstm = tf.keras.layers.LSTM(1)

lstm(input)



lstm.cell.kernel = tf.Variable(np.random.randn(115, 4))

yea = tf.keras.Sequential([tf.keras.layers.LSTM(64, return_sequences=True), tf.keras.layers.LSTM(64), tf.keras.layers.Dense(6)])