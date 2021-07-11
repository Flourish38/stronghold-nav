@time begin
    println("Loading packages...")
    using Flux
    using BSON
    using BSON: @load
    using ONNXmutable
end

println("yea")

bson("prayer.bson", BSON.parse("models/rl_recurrent_d_3111.bson"))
BSON.load("prayer.bson")

begin
    oh_no = BSON.parse("models/rl_recurrent_d_3111.bson")
    lstm_1 = oh_no[:rl_model][:data][1][:data][1][:data][2]
end

begin
    oh_ok = BSON.parse("test.bson")
    lstm_a = oh_ok[:yea][:data][1][:data][1][:data][1]
end

function load_broken_model(path)
    brok = BSON.parse(path)
    layers = []
    for i in 1:2
        lstm = brok[:rl_model][:data][1][:data][i][:data][1]
        ps = BSON.raise_recursive(lstm[:data], Main)
        ps[4] = reshape(ps[4], :, 1)
        ps[5] = reshape(ps[5], :, 1)
        push!(layers, Flux.Recur(Flux.LSTMCell(ps[1], ps[2], ps[3], (ps[4], ps[5]))))
    end
    push!(layers, BSON.raise_recursive(brok[:rl_model][:data][1][:data][3], Main))
    if haskey(brok, :iters)
        return Dict(:iters => brok[:iters], :rl_model => Chain(layers...), :adam => BSON.raise_recursive(brok[:adam], Main))
    end
    return Dict(:rl_model => Chain(layers...), :adam => BSON.raise_recursive(brok[:adam], Main))
end

begin
    path = "models/tmp/"
    for fname in readdir(path)
        bson(path*"fixed_"*fname, load_broken_model(path*fname))
    end
    #bson("models/fixed_rl_recurrent_d_3111.bson", load_broken_model("models/rl_recurrent_d_3111.bson"))
end

begin
    model = BSON.load("models/fixed_rl_recurrent_d_3111.bson")[:rl_model]
    onnx("test.onnx", model)
end