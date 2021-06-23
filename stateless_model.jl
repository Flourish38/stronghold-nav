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
    @assert 0 < room <= length(stronghold)
    r = stronghold[room]
    pieces = vcat([r.piece, get_piece(stronghold, r.parent)], [get_piece(stronghold, exit) for exit in r.exits])
    return vcat([r.depth], onehot(r.orientation, 1:4), [onehot(p, 1:14) for p in pieces]...)
end

function encode_stronghold(s)
    X = [encode_room(s, r) for r in eachindex(s)]
    Y = [onehot(s[r].correctDirection, 0:5) for r in eachindex(s)]
    return X, Y
end

function navigates(s, model)
    n = length(s)
    r = s[n]
    while r.piece != 11
        exit = argmax(model(encode_room(s, n))[end-4:end])
        if exit != r.correctDirection
            return false
        end
        n = r.exits[exit]
        r = s[n]
    end
    return true
end

begin
    model = Chain(Dense(103, 20, relu), Dense(20, 8, relu), Dense(8, 6))
    loss(x::AbstractVector{<:AbstractVector}, y::AbstractVector) = sum(loss.(x, y))
    loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
    opt = ADAM()
    data = (encode_stronghold(s) for s in strongholds[train])
end

function train_cb()
    score = mean([navigates(s, model) for s in strongholds[dev]])
    println(Time(now()), "\t", score)
    score > 0.195 && Flux.stop()
end

begin
    @time Flux.train!(loss, params(model), data, opt; cb = Flux.throttle(train_cb, 30))
end

begin
    navigates_test = mean([navigates(s, model) for s in strongholds[test]])
end

begin
    bson("models/stateless_185.bson", stateless_model = model)
end
