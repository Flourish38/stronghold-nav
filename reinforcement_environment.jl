begin
    using StaticArrays
    using Reinforce
end

begin  # Necessary for goto
if @isdefined ENVIRONMENT_DECLARED
    @goto skip_environment_declare
end

# INPUT_VEC_ORDER must be defined before including this file
println(INPUT_VEC_ORDER)

include("load_strongholds.jl")

begin  # Parameters and rewards
    MAX_STEPS = 50

    TARGET_ROOM = Int8(50)
    CORRECT_DIRECTION = Int8(1)
    WRONG_DIRECTION = Int8(-1)
    STUPID_ROOM = Int8(-20)
    CLOSED_EXIT = Int8(-10)
    LOOPING_BEHAVIOR = Int8(-100)
    INVALID_EXIT = Int8(-15)
end

begin
    const INPUT_VEC_FORMAT = split(INPUT_VEC_ORDER, ',')
    const STATE_WIDTHS = Dict(
        "CURRENT" => 14,
        "PARENT_ROOM" => 14,
        "PREVIOUS_ROOM" => 14,
        "EXIT_1" => 14,
        "EXIT_2" => 14,
        "EXIT_3" => 14,
        "EXIT_4" => 14,
        "EXIT_5" => 14,
        "PREV_EXIT_INDEX" => 1,
        "PREV_EXIT_INDEX_COMPAT" => 6,
        "DIRECTION" => 4,
        "DEPTH" => 1,
        "CONSTANT" => 1,
        "DOWNWARDS" => 1,
        "ENTRY" => 6
    )
    const STATE_WIDTH = sum(STATE_WIDTHS[x] for x in INPUT_VEC_FORMAT)
end

struct StrongholdReplay{N, N1}  # boo, and ComputedFieldTypes didn't work (N1 is just N + 1)
    states::MMatrix{STATE_WIDTH, N1, UInt8}
    actions::MVector{N, Int8}
    rewards::MVector{N, Int8}
    function StrongholdReplay(n)
        replay = new{n, n+1}(MMatrix{STATE_WIDTH, n+1, UInt8}(undef), MVector{n, Int8}(undef), MVector{n, Int8}(undef))
        fill!(replay.actions, Int8(-1))
        replay
    end
    StrongholdReplay() = StrongholdReplay(MAX_STEPS)
end

function Base.iterate(replay::StrongholdReplay{N}, i=1) where N
    (i > N || replay.actions[i] == -1) && return nothing
    return (replay.states[:, i], replay.actions[i], replay.rewards[i], replay.states[:, i+1]), i + 1
end

function Base.length(replay::StrongholdReplay{N}) where N 
    replay.actions[N] != Int8(-1) && (return N)
    return searchsortedlast(replay.actions, Int8(-1), lt=(>=))  # a little magic for ya (I didn't want to write binary search myself)
end

function Base.size(replay::StrongholdReplay{N}) where N
    return N
end

function valid_data(replay, n)
    return states(replay, n), actions(replay, n), rewards(replay, n)
end
valid_data(replay) = valid_data(replay, length(replay))

states(replay, n) = view(replay.states, :, SOneTo(n+1))  # For some reason, if you use indexing instead of a view, it takes QUADRATIC COMPILE TIME??? No thank you.
actions(replay, n) = view(replay.actions, SOneTo(n))
rewards(replay, n) = view(replay.rewards, SOneTo(n))

states(replay) = states(replay, length(replay))
actions(replay) = actions(replay, length(replay))
rewards(replay) = rewards(replay, length(replay))


mutable struct StrongholdEnvironment <: AbstractEnvironment
    resettable::Bool
    reward::Int8
    steps::Int
    replay::StrongholdReplay
    visited_rooms::BitVector
    stronghold::Vector{Room}
    room::Int16
    last_piece::UInt8
    last_exit::UInt8
    entry::UInt8
    target_piece::UInt8
    function StrongholdEnvironment(s::Vector{Room}, r::Bool=true, target_piece::UInt8=UInt8(11))
        # This assumes you start in the portal room if you want to go anywhere else (pre-1.9 mode)
        room::Int16 = target_piece == 11 ? length(s) : findfirst(x -> x.piece == 11, s)
        env = new(r, 0, 0, StrongholdReplay(), falses(length(s)), s, room, 14, 0, 0, target_piece)
        set_state!(env)
        env
    end
end

function StrongholdEnvironment(r::Bool=true, target_piece::UInt8=UInt8(11))
    return StrongholdEnvironment(strongholds[rand(train)], r, target_piece)
end

begin  # Optimization hell
    get_piece(env::StrongholdEnvironment, room)::UInt8 = get_piece(env.stronghold, room)
    function get_piece(stronghold, room)::UInt8
        return room <= 0 ? 14 : stronghold[room].piece
    end
    # These all return an integer. If it is a scalar function, it returns the scalar. If it is a vector function, it returns the onehot index (1-indexed.)
    const STATE_FUNCTIONS = Dict(
        "CURRENT" => (env::StrongholdEnvironment, c::Room) -> c.piece,
        "PARENT_ROOM" => (env::StrongholdEnvironment, c::Room) -> get_piece(env, c.parent),
        "PREVIOUS_ROOM" => (env::StrongholdEnvironment, c::Room) -> env.last_piece,
        "EXIT_1" => (env::StrongholdEnvironment, c::Room) -> get_piece(env, c.exits[1]),
        "EXIT_2" => (env::StrongholdEnvironment, c::Room) -> get_piece(env, c.exits[2]),
        "EXIT_3" => (env::StrongholdEnvironment, c::Room) -> get_piece(env, c.exits[3]),
        "EXIT_4" => (env::StrongholdEnvironment, c::Room) -> get_piece(env, c.exits[4]),
        "EXIT_5" => (env::StrongholdEnvironment, c::Room) -> get_piece(env, c.exits[5]),
        "PREV_EXIT_INDEX" => (env::StrongholdEnvironment, c::Room) -> env.last_exit,
        "PREV_EXIT_INDEX_COMPAT" => (env::StrongholdEnvironment, c::Room) -> env.last_exit + one(UInt8),
        "DIRECTION" => (env::StrongholdEnvironment, c::Room) -> c.orientation,
        "DEPTH" => (env::StrongholdEnvironment, c::Room) -> c.depth,
        "CONSTANT" => (env::StrongholdEnvironment, c::Room) -> zero(UInt8),
        "DOWNWARDS" => (env::StrongholdEnvironment, c::Room) -> UInt8(env.entry == 0),
        "ENTRY" => (env::StrongholdEnvironment, c::Room) -> env.entry + one(UInt8)
    )
    const NUM_SCALAR_FUNCTIONS = sum(STATE_WIDTHS[x] == 1 for x in INPUT_VEC_FORMAT)
    const NUM_VECTOR_FUNCTIONS = length(INPUT_VEC_FORMAT) - NUM_SCALAR_FUNCTIONS
    # Collect the functions for fast reference later
    const SCALAR_FUNCTION_LIST = SVector{NUM_SCALAR_FUNCTIONS}([STATE_FUNCTIONS[x] for x in INPUT_VEC_FORMAT if STATE_WIDTHS[x] == 1])
    const VECTOR_FUNCTION_LIST = SVector{NUM_VECTOR_FUNCTIONS}([STATE_FUNCTIONS[x] for x in INPUT_VEC_FORMAT if STATE_WIDTHS[x] != 1])
    # Trust me on these
    # Makes a Vector of indices for where to put the results of the scalar functions
    const SCALAR_FUNCTION_OFFSETS = SVector{NUM_SCALAR_FUNCTIONS}([sum([STATE_WIDTHS[x] for x in INPUT_VEC_FORMAT[1:i]]) + 1 for i in 0:(length(INPUT_VEC_FORMAT)-1) if STATE_WIDTHS[INPUT_VEC_FORMAT[i+1]] == 1])
    # Makes a Vector of indices ONE BELOW the start of the one-hot vector in the output.
    # Since the vector functions return a 1-indexed scalar, if you add 1 to the vector function offset you should get the first index of the output vector.
    const VECTOR_FUNCTION_OFFSETS = SVector{NUM_VECTOR_FUNCTIONS}([sum([STATE_WIDTHS[x] for x in INPUT_VEC_FORMAT[1:i]]) for i in 0:(length(INPUT_VEC_FORMAT)-1) if STATE_WIDTHS[INPUT_VEC_FORMAT[i+1]] != 1])
end

begin
    current(env::StrongholdEnvironment)::Room = env.stronghold[env.room]
    function get_state(env)  # This runs in <100ns :)
        c = current(env)
        output = @MVector zeros(UInt8, STATE_WIDTH)
        output[SCALAR_FUNCTION_OFFSETS] .= SVector{NUM_SCALAR_FUNCTIONS}(x(env, c) for x in SCALAR_FUNCTION_LIST)
        output[VECTOR_FUNCTION_OFFSETS .+ SVector{NUM_VECTOR_FUNCTIONS}(x(env, c) for x in VECTOR_FUNCTION_LIST)] .= 1
        return output
    end
    # Despite literally rearranging the entire codebase so I could do in-place mutates... they weren't faster. compiler couldn't figure it out.
    function set_state!(env)  
        env.replay.states[:, env.steps+1] = get_state(env)
        return  # saves like 20 ns to not return :P
    end
end

begin
    Reinforce.finished(env::StrongholdEnvironment, s′) = current(env).piece == env.target_piece
    Reinforce.actions(env::StrongholdEnvironment, s) = Int8(0):Int8(5)
    Reinforce.state(env::StrongholdEnvironment) = view(env.replay.states, :, env.steps+1)
    function Reinforce.reset!(env::StrongholdEnvironment)
        env.resettable || return env
        env.stronghold = strongholds[rand(train)]
        env.room = env.target_piece == 11 ? length(env.stronghold) : findfirst(x -> x.piece == 11, env.stronghold)
        env.visited_rooms = falses(length(env.stronghold))
        env.last_piece = 14
        env.last_exit = 0
        env.entry = 0
        env.reward = 0
        env.steps = 0
        env.replay = StrongholdReplay()
        return env
    end
    function Reinforce.step!(env::StrongholdEnvironment, s, a)
        env.steps += 1
        c = current(env)
        if a > PIECE_TO_NUM_EXITS[c.piece] || a == 0 && c.piece == 13  # Chose an exit that can never have a room
            env.reward = INVALID_EXIT
            env.replay.states[:, env.steps+1] = env.replay.states[:, env.steps]
        elseif a != 0 && c.exits[a] <= 0  # Chose an exit that doesn't have a room
            env.reward = CLOSED_EXIT
            env.replay.states[:, env.steps+1] = env.replay.states[:, env.steps]
        else
            # If you have already been in this room, and you go back the way you came, you're looping.
            if env.visited_rooms[env.room] && a == env.entry 
                env.reward = LOOPING_BEHAVIOR
            else
                env.reward = c.targetDirection == a ? CORRECT_DIRECTION : WRONG_DIRECTION
            end
            env.last_piece = c.piece
            env.last_exit = a
            env.visited_rooms[env.room] = true
            if a == 0  # Going towards the start requires a search
                env.entry = findfirst(==(env.room), env.stronghold[c.parent].exits)
                env.room = c.parent
            else
                env.room = c.exits[a]
                env.entry = 0
            end
            set_state!(env)
        end
        c = current(env)
        if c.piece == env.target_piece  # Success! Much reward.
            env.reward = TARGET_ROOM
        elseif c.piece > 9  # It happens that this is all of the stupid rooms (rooms that never have an exit other than the entrance)
            env.reward = min(env.reward, STUPID_ROOM)
        end
        env.replay.rewards[env.steps] = reward(env)
        env.replay.actions[env.steps] = a
        return reward(env), state(env)
    end
    Reinforce.ismdp(::StrongholdEnvironment) = false
    Reinforce.maxsteps(::StrongholdEnvironment) = size(env.replay)
end

ENVIRONMENT_DECLARED = true

@label skip_environment_declare
end
return