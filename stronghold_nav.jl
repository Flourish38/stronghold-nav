@time begin
    using ProgressMeter
    using BenchmarkTools: @benchmark
    using Statistics
    using Flux
    using Flux: onehot, onecold
    using Dates
    using StaticArrays
    using BSON
end

begin
    orientations_vec = ["E", "W", "N", "S"]
    orientations = Dict{String, Int8}(orientations_vec[i] => i for i in eachindex(orientations_vec))
    pieces_vec = ["Corridor", "PrisonHall", "LeftTurn", "RightTurn", "SquareRoom", "Stairs", "SpiralStaircase", "FiveWayCrossing", "ChestCorridor", "Library", "PortalRoom", "SmallCorridor", "Start", "None"]
    pieces = Dict{String, Int8}(pieces_vec[i] => i for i in eachindex(pieces_vec))
    piece_to_num_exits = [3, 1, 1, 1, 3, 1, 1, 5, 1, 0, 0, 0, 1, 0]
end

mutable struct Room
    exits::SVector{5, Int16}
    piece::Int8
    orientation::Int8
    parent::Int16
    depth::Int8
    correctDirection::Int8
    function Room(s::String)
        info = split(s)
        exits = (x -> parse(Int16, x)+1).(info[3:end])
        exits = rpad(exits, 5, -1)
        new(exits, pieces[info[1]], orientations[info[2]], -1, -1, -1)
    end
end

valid_exits(r::Room) = r.exits[1:piece_to_num_exits[r.piece]]

function assignParents!(stronghold)
    queue = Set{Int}()
    n = length(stronghold)
    r = stronghold[n]
    @assert r.piece == 13
    r.parent = 0
    r.depth = 0
    push!(queue, n)
    while !isempty(queue)
        n = pop!(queue)
        r = stronghold[n]
        for i in r.exits
            if i > 0
                stronghold[i].parent = n
                stronghold[i].depth = r.depth + 1
                push!(queue, i)
            end
        end
    end
end

function assignCorrectDirection!(stronghold)
    queue = Set{Int}()
    n = findfirst(r -> r.piece == 11, stronghold)
    stronghold[n].correctDirection = 0
    push!(queue, n)
    while !isempty(queue)
        n = pop!(queue)
        r = stronghold[n]
        i = r.parent
        if i > 0 && stronghold[i].correctDirection == -1
            stronghold[i].correctDirection = findfirst(==(n), stronghold[i].exits)
            push!(queue, i)
        end
        for i in r.exits
            if i > 0 && stronghold[i].correctDirection == -1
                stronghold[i].correctDirection = 0
                push!(queue, i)
            end
        end
    end
end

@time begin
    strongholds = Vector{Room}[]
    @time open("data/outputDirections.txt", "r") do io
        line = ""
        #p = Progress(100000; showspeed=true)
        while true
            stronghold = Room[]
            while !occursin("START", line)
                line = readline(io)
                eof(io) && break
            end
            eof(io) && break
            readline(io)
            while true
                line = readline(io)
                occursin("END", line) && break
                push!(stronghold, Room(line))
            end
            push!(strongholds, stronghold)
            #next!(p)
        end
    end
    @time assignParents!.(strongholds)
    @time assignCorrectDirection!.(strongholds)
end

function get_piece(stronghold, room)::Int8
    return room <= 0 ? 14 : stronghold[room].piece
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

function print_correct_path(stronghold)
    r = stronghold[end]
    while r.piece != 11
        println(r)
        r = stronghold[r.exits[r.correctDirection]]
    end
    println(r)
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

rec_sizeof(a::AbstractArray) = sizeof(a) + sum(rec_sizeof.(a))
rec_sizeof(a) = sizeof(a)

begin
    model = Chain(Dense(103, 20, relu), Dense(20, 8, relu), Dense(8, 6))
    loss(x::AbstractVector{<:AbstractVector}, y::AbstractVector) = sum(loss.(x, y))
    loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
    opt = ADAM()
    data = (encode_stronghold(s) for s in strongholds[1:90000])
    @show mean([navigates(s, model) for s in strongholds[90001:95000]])
end

function train_cb()
    score = mean([navigates(s, model) for s in strongholds[90001:95000]])
    println(Time(now()), "\t", score)
    score > 0.195 && Flux.stop()
end

begin
    @time Flux.train!(loss, params(model), data, opt; cb = Flux.throttle(train_cb, 30))
end

begin
    navigates_test = mean([navigates(s, model) for s in strongholds[95001:100000]])
end

begin
    bson("models/stateless_185.bson", stateless_model = model)
end
