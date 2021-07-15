import pickle
import tensorflow as tf
import numpy as np

def load_pickle(path):
    obj = None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

lstm1 = load_pickle("models/tmp/lstm1.pickle")
lstm2 = load_pickle("models/tmp/lstm2.pickle")
dense = load_pickle("models/tmp/dense.pickle")

class ISLSTM(tf.keras.layers.LSTM):
    def __init__(self, in_units, units, initial_states, return_sequences=False):
        super(ISLSTM, self).__init__(units, return_sequences=return_sequences, batch_input_shape=(None, None, in_units))
        self.initial_states = [tf.constant(s) for s in initial_states]
    
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(ISLSTM, self).call(inputs, mask=mask, training=training, initial_state=self.initial_states)


def lstm_to_islstm(dict, ret):
    input = tf.ones((1, 1, dict["Wi"].shape[0]))
    lstm = ISLSTM(dict["Wi"].shape[0], dict["Wh"].shape[0], dict["state0"], return_sequences=ret)
    lstm(input)
    lstm.cell.kernel = tf.Variable(dict["Wi"])
    lstm.cell.recurrent_kernel = tf.Variable(dict["Wh"])
    lstm.cell.bias = tf.Variable(dict["b"])
    return lstm


def dense_to_tf(dict):
    input = tf.ones((1, dict["weight"].shape[0]))
    dense = tf.keras.layers.Dense(len(dict["bias"]))
    dense(input)
    dense.kernel = tf.Variable(dict["weight"])
    dense.bias = tf.Variable(dict["bias"])
    return dense


model = tf.keras.Sequential([
    lstm_to_islstm(lstm1, True),
    lstm_to_islstm(lstm2, False),
    dense_to_tf(dense)
])

input = np.random.randn(1, 1, 115)
model(input)
model.reset_states()
model.save("models/rl_rnn_2/model")
