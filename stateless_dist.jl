@time include("load_strongholds.jl")

@time begin
    println("Loading packages...")
    using Statistics
    using Flux
    using Flux: onehot, onecold
    using Dates
    using BSON
end

function encode_room(stronghold, room)
    r = stronghold[room]
    pieces = vcat([r.piece, get_piece(stronghold, r.parent)], [get_piece(stronghold, exit) for exit in r.exits])
    return vcat([r.depth], onehot(r.orientation, 1:4), [onehot(p, 1:14) for p in pieces]...)
end

function encode_output(stronghold, room)
    r = stronghold[room]
    output = zeros(Int8, 64)
    d = r.correctDirection
    if d == 0
        dist = stronghold[r.parent].targetDistance
        dist = clamp(dist, 1, 24)
        output[dist] = 1
    else
        dist = stronghold[r.exits[d]].targetDistance
        dist = clamp(dist, 1, 8)
        output[24 + 8(d-1) + dist] = 1
    end
    return output
end

function encode_stronghold(s)
    X = [encode_room(s, r) for r in eachindex(s)]
    Y = [encode_output(s, r) for r in eachindex(s)]
    return X, Y
end

function navigates(s, model, verbose=false)
    n = length(s)
    r = s[n]
    while r.piece != 11
        out = softmax(model(encode_room(s, n)))
        probs = [sum(out[(25+8i):(32+8i)]) for i in 0:4]
        if verbose
            for i in 0:7
                println(out[(8i+1):(8i+8)])
            end
            println(sum(out[1:24]), "\t", probs, "\t", r.correctDirection)
            println(sum(out[1:24] ./ (1:24)), "\t", [sum(out[(25+8i):(32+8i)] ./ (1:8)) for i in 0:4], "\t", argmax(probs), "\n")
        end
        exit = argmax(probs)
        if exit != r.correctDirection
            return false
        end
        n = r.exits[exit]
        r = s[n]
    end
    return true
end

begin
    model = Chain(Dense(103, 64, relu), Dense(64, 64, relu), Dense(64, 64))
    loss(x::AbstractVector{<:AbstractVector}, y::AbstractVector) = sum(loss.(x, y))
    loss(x, y) = Flux.logitcrossentropy(model(x), y)
    opt = ADAM()
    data = (encode_stronghold(s) for s in strongholds[train])
    nothing  # repl-friendly
end

function train_cb()
    score = mean([navigates(s, model) for s in strongholds[dev]])
    println(Time(now()), "\t", score)
    score >= 0.195 && Flux.stop()
end

begin
    @time Flux.train!(loss, params(model), data, opt; cb = Flux.throttle(train_cb, 30))
end

begin
    navigates_test = mean([navigates(s, model) for s in strongholds[test]])
end

begin
    bson("models/stateless_dist_188.bson", stateless_dist = model)
end

begin
    const INPUT_VEC_ORDER = "DEPTH,DIRECTION,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5" # DEPTH,
    @time include("reinforcement_environment.jl")
    include("utils.jl")
end

p = QApproximatorPolicy(model, nothing, nothing)
begin
    env = StrongholdEnvironment(strongholds[dev[1]], false)
    for (s, a, r, s′) in Episode(env, p)
        println(model(s), "\t", a, "\t", r)
    end
end