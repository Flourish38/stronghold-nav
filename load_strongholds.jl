@time begin
    println("Loading packages...")
    using StaticArrays
end

begin
    const orientations_vec = ["E", "W", "N", "S"]
    const orientations = Dict{String, Int8}(orientations_vec[i] => i for i in eachindex(orientations_vec))
    const pieces_vec = ["Corridor", "PrisonHall", "LeftTurn", "RightTurn", "SquareRoom", "Stairs", "SpiralStaircase", "FiveWayCrossing", "ChestCorridor", "Library", "PortalRoom", "SmallCorridor", "Start", "None"]
    const pieces = Dict{String, Int8}(pieces_vec[i] => i for i in eachindex(pieces_vec))
    const piece_to_num_exits = [3, 1, 1, 1, 3, 1, 1, 5, 1, 0, 0, 0, 1, 0]
end

mutable struct Room  # Very space-optimized; must be mutable to assign parent, depth, and correctDirection later.
    exits::SVector{5, Int16}
    piece::Int8
    orientation::Int8
    parent::Int16
    depth::Int8
    correctDirection::Int8
    function Room(s::String)
        info = split(s)
        exits = (x -> parse(Int16, x)+1).(info[3:end])
        exits = vcat(exits, fill(-1, 5 - length(exits)))
        new(exits, pieces[info[1]], orientations[info[2]], -1, -1, -1)
    end
end

valid_exits(r::Room) = vcat(r.parent > 0 ? [0] : [], (1:5)[r.exits .> 0])

function assignParents!(stronghold)  # Standard tree flood from root algorithm
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

function assignCorrectDirection!(stronghold)  # Tree flood, this time from portal room (leaf)
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

rec_sizeof(a::AbstractArray) = sizeof(a) + sum(rec_sizeof.(a))
rec_sizeof(a) = sizeof(a)

function load_strongholds()
    println("Loading Strongholds...")
    strongholds = Vector{Room}[]
    open("data/outputDirections.txt", "r") do io
        line = ""
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
        end
    end
    assignParents!.(strongholds)
    assignCorrectDirection!.(strongholds)
    return strongholds
end

begin
    strongholds = @time load_strongholds()
    train = 1:90000
    dev = 90001:95000
    test = 95001:100000
end

function get_piece(stronghold, room)::Int8
    return room <= 0 ? 14 : stronghold[room].piece
end

function print_correct_path(stronghold)
    r = stronghold[end]
    while r.piece != 11
        println(r)
        r = stronghold[r.exits[r.correctDirection]]
    end
    println(r)
end

# This makes the repl happy when you include the file
return
