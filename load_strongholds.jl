begin
    using StaticArrays
    using ProgressMeter
end

begin
    const _orientations_vec = @SVector ["E", "W", "N", "S"]
    const ORIENTATIONS = Dict{String, Int8}(x => i for (i, x) in enumerate(_orientations_vec))
    const _pieces_vec = @SVector ["Corridor", "PrisonHall", "LeftTurn", "RightTurn", "SquareRoom", "Stairs", "SpiralStaircase", "FiveWayCrossing", "ChestCorridor", "Library", "PortalRoom", "SmallCorridor", "Start", "None"]
    const PIECES = Dict{String, Int8}(x => i for (i, x) in enumerate(_pieces_vec))
    const PIECE_TO_NUM_EXITS = @SVector [3, 1, 1, 1, 3, 1, 1, 5, 1, 0, 0, 0, 1, 0]
end

mutable struct Room  # Very space-optimized; must be mutable to assign parent, depth, and target details later.
    exits::SVector{5, Int16}
    piece::UInt8
    orientation::UInt8
    parent::Int16
    depth::UInt8
    targetDirection::UInt8
    targetDistance::UInt8
    function Room(s::String)
        info = split(s)
        exits = (x -> parse(Int16, x)+1).(info[3:end])
        exits = vcat(exits, fill(-1, 5 - length(exits)))
        new(exits, PIECES[info[1]], ORIENTATIONS[info[2]], 0, 0, 255, 0)  # targetDirection must be >5 (a nonsense value)
    end
end

function assignParents!(stronghold)  # Standard tree flood from root algorithm
    queue = Set{UInt16}()
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
    queue = Set{UInt16}()
    n = findfirst(r -> r.piece == 11, stronghold)
    stronghold[n].targetDirection = 0
    stronghold[n].targetDistance = 0
    push!(queue, n)
    while !isempty(queue)
        n = pop!(queue)
        r = stronghold[n]
        i = r.parent
        if i > 0 && stronghold[i].targetDirection > 5  # targetDirection > 5 is a nonsense value used for initialization
            stronghold[i].targetDirection = findfirst(==(n), stronghold[i].exits)
            stronghold[i].targetDistance = r.targetDistance + 1
            push!(queue, i)
        end
        for i in r.exits
            if i > 0 && stronghold[i].targetDirection > 5
                stronghold[i].targetDirection = 0
                stronghold[i].targetDistance = r.targetDistance + 1
                push!(queue, i)
            end
        end
    end
end

function load_strongholds(filename)
    println("Loading Strongholds...")
    strongholds = Vector{Room}[]
    open(filename, "r") do io
        seekend(io)
        filesize = position(io)
        p = Progress(filesize; dt=1, showspeed=true)
        seekstart(io)
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
            update!(p, position(io))
        end
    end
    assignParents!.(strongholds)
    assignCorrectDirection!.(strongholds)
    return strongholds
end

begin
    strongholds = load_strongholds("data/outputDirections.txt")
    n = length(strongholds)
    train = 1:floor(Int, 0.9n)
    dev = (floor(Int, 0.9n)+1):floor(Int, 0.95n)
    test = (floor(Int, 0.95n)+1):n
end

# This makes the repl happy when you include the file
return
