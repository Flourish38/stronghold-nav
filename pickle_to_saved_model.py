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

# Initial State LSTM
class ISLSTM(tf.keras.layers.LSTM):
    def __init__(self, in_units, units, initial_states, return_sequences=False):
        super(ISLSTM, self).__init__(units, return_sequences=return_sequences, batch_input_shape=(None, None, in_units))
        # Conveniently, providing the default LSTM with initial state is similar to in julia; it just takes a vector of matrices
        self.initial_states = [tf.constant(s) for s in initial_states]
    
    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(ISLSTM, self).call(inputs, mask=mask, training=training, initial_state=self.initial_states)


def lstm_to_islstm(dict, ret):
    input = tf.ones((1, 1, dict["Wi"].shape[0]))
    # The "return_sequences" variable allows for the model to function with batched input.
    # All recurrent layers should have return_sequences set to True EXCEPT for the last one, which should be False.
    # This ensures compatibility with the Stronghold Trainer mod.
    # If you would rather pass input one at a time, all should be set to True.
    lstm = ISLSTM(dict["Wi"].shape[0], dict["Wh"].shape[0], dict["state0"], return_sequences=ret)
    lstm(input)  # "compile" the model
    lstm.cell.kernel = tf.Variable(dict["Wi"])
    lstm.cell.recurrent_kernel = tf.Variable(dict["Wh"])
    lstm.cell.bias = tf.Variable(dict["b"])
    return lstm


def dense_to_tf(dict):
    input = tf.ones((1, dict["weight"].shape[0]))
    dense = tf.keras.layers.Dense(len(dict["bias"]))
    dense(input)  # "compile" the model
    dense.kernel = tf.Variable(dict["weight"])
    dense.bias = tf.Variable(dict["bias"])
    return dense


model = tf.keras.Sequential([
    lstm_to_islstm(lstm1, True),
    lstm_to_islstm(lstm2, False),
    dense_to_tf(dense)
])

input = np.random.randn(1, 1, 115)
model(input)  # "compile" the model, which is necessary to save it
model.reset_states()
model.save("models/rl_rnn_2.3/model")
