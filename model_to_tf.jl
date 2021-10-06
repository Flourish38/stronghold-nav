@time begin
    using Conda
    using PyCall
    using BSON
    using Flux
    tf = pyimport("tensorflow")
    copy = pyimport("copy")
    pickle = pyimport("pickle")
end

# Doesn't work. Since tensorflow LSTMs by default do not support custom initial state,
# a custom class is required, and doing that through PyCall is too much work.
# It is easier to pickle the matrices and handle things on the python side.
function lstm_to_tensorflow(lstm_cell, ret)
    input = ones(1, 1, size(lstm_cell.Wi, 2))
    lstm = tf.keras.layers.LSTM(size(lstm_cell.Wh, 2), return_sequences=ret, stateful=true)
    lstm(input)
    lstm.cell.kernel = tf.Variable(transpose(lstm_cell.Wi))
    lstm.cell.recurrent_kernel = tf.Variable(transpose(lstm_cell.Wh))
    lstm.cell.bias = tf.Variable(lstm_cell.b)
    lstm.initial_states = tf.constant.(transpose.(lstm_cell.state0))
    py"""
    def reset_states(self, states=None):
        self.states = $(tf.Variable.(transpose.(lstm_cell.state0)))
    """
    reset_states = py"reset_states"
    lstm.reset_states = reset_states.__get__(lstm, tf.keras.layers.LSTM)
    return lstm
end

# Basic general pickler, should work with most cell types but definitely works with Dense and LSTM
function cell_to_pickle(cell, path)
    output = Dict()
    for fn in fieldnames(typeof(cell))
        x = getfield(cell, fn)
        # Have to transpose the matrices because julia is column-first and python is row-first
        if typeof(x) <: AbstractMatrix
            output[String(fn)] = transpose(x)
        elseif typeof(x) <: Union{AbstractVector, Tuple}  # LSTMs have 2 state variables, they are stored in one field
            output[String(fn)] = transpose.(x)
        end
    end
    @pywith pybuiltin("open")(path, "wb") as f begin
        pickle.dump(output, f)
    end
end

# If your model is not recurrent, you can just use this
function dense_to_tensorflow(dense)
    input = ones(1, size(dense.W, 2))
    fc = tf.keras.layers.Dense(size(dense.W, 1))
    fc(input)
    fc.kernel = tf.constant(transpose(dense.W))
    fc.bias = tf.constant(dense.b)
    return fc
end

begin  # save non-recurrent model
    saving_model = BSON.load("models/tmp/rl_recurrent_e_131.bson")[:rl_model]
    tf_model = tf.keras.Sequential([
        dense_to_tensorflow(saving_model[1]),
        dense_to_tensorflow(saving_model[2]),
        dense_to_tensorflow(saving_model[3])
    ])
    
    input = rand(1, 115)
    tf_model(input)  # This "compiles" the model, which is necessary in order to save it

    tf_model.save("models/rl_stateless_3/model")
end

begin  # pickle recurrent model for processing in pickle_to_saved_model.py
    saving_model = BSON.load("models/tmp/rl_rnn_a2_19406.bson")[:rl_model]
    cell_to_pickle(saving_model[1].cell, "models/tmp/lstm1.pickle")
    cell_to_pickle(saving_model[2].cell, "models/tmp/lstm2.pickle")
    cell_to_pickle(saving_model[3], "models/tmp/dense.pickle")
end