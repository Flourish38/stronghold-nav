println(INPUT_VEC_ORDER)

include("load_strongholds.jl")

@time begin
    println("Loading packages...")
    using StaticArrays
    using Reinforce
end

begin
    const MAX_STEPS = 100
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

mutable struct StrongholdEnvironment <: AbstractEnvironment
    resettable::Bool
    reward::Float64
    steps::Int
    state::SVector{STATE_WIDTH, Float32}
    stronghold::Vector{Room}
    room::Int16
    last_piece::Int8
    last_exit::Int8
    entry::Int8
    function StrongholdEnvironment(s::Vector{Room}, r::Bool = true)
        env = new(r, 0, 0, (@SVector zeros(Float32, STATE_WIDTH)), s, length(s), 14, 0, 0)
        env.state = get_state(env)
        env
    end
end

function StrongholdEnvironment(r::Bool = true)
    return StrongholdEnvironment(strongholds[rand(train)], r)
end

begin
    current(env::StrongholdEnvironment)::Room = env.stronghold[env.room]
    function get_state(env::StrongholdEnvironment)  # This runs in <100ns :)
        c = current(env)
        output = @MVector zeros(Int8, STATE_WIDTH)
        output[SCALAR_FUNCTION_OFFSETS] .= SVector{NUM_SCALAR_FUNCTIONS, Int8}(x(env, c) for x in SCALAR_FUNCTION_LIST)
        output[VECTOR_FUNCTION_OFFSETS .+ SVector{NUM_VECTOR_FUNCTIONS, Int8}(x(env, c) for x in VECTOR_FUNCTION_LIST)] .= 1
        return output
    end
end


begin  # Optimization hell
    const PORTAL_ROOM = 50.
    const CORRECT_DIRECTION = -1.
    const WRONG_DIRECTION = -1.
    const STUPID_ROOM = -50.
    const CLOSED_EXIT = -50.
    const INVALID_EXIT = -100.
    # These all return an integer. If it is a scalar function, it returns the scalar. If it is a vector function, it returns the onehot index (1-indexed.)
    const STATE_FUNCTIONS = Dict(
        "CURRENT" => (env::StrongholdEnvironment, c::Room) -> c.piece,
        "PARENT_ROOM" => (env::StrongholdEnvironment, c::Room) -> get_piece(env.stronghold, c.parent),
        "PREVIOUS_ROOM" => (env::StrongholdEnvironment, c::Room) -> env.last_piece,
        "EXIT_1" => (env::StrongholdEnvironment, c::Room) -> get_piece(env.stronghold, c.exits[1]),
        "EXIT_2" => (env::StrongholdEnvironment, c::Room) -> get_piece(env.stronghold, c.exits[2]),
        "EXIT_3" => (env::StrongholdEnvironment, c::Room) -> get_piece(env.stronghold, c.exits[3]),
        "EXIT_4" => (env::StrongholdEnvironment, c::Room) -> get_piece(env.stronghold, c.exits[4]),
        "EXIT_5" => (env::StrongholdEnvironment, c::Room) -> get_piece(env.stronghold, c.exits[5]),
        "PREV_EXIT_INDEX" => (env::StrongholdEnvironment, c::Room) -> env.last_exit,
        "PREV_EXIT_INDEX_COMPAT" => (env::StrongholdEnvironment, c::Room) -> env.last_exit + 1,
        "DIRECTION" => (env::StrongholdEnvironment, c::Room) -> c.orientation,
        "DEPTH" => (env::StrongholdEnvironment, c::Room) -> c.depth,
        "CONSTANT" => (env::StrongholdEnvironment, c::Room) -> 0,
        "DOWNWARDS" => (env::StrongholdEnvironment, c::Room) -> env.entry == 0,
        "ENTRY" => (env::StrongholdEnvironment, c::Room) -> env.entry + 1
    )
    const NUM_SCALAR_FUNCTIONS = sum(STATE_WIDTHS[x] == 1 for x in INPUT_VEC_FORMAT)
    const NUM_VECTOR_FUNCTIONS = length(INPUT_VEC_FORMAT) - NUM_SCALAR_FUNCTIONS
    const SCALAR_FUNCTION_LIST = SVector{NUM_SCALAR_FUNCTIONS}([STATE_FUNCTIONS[x] for x in INPUT_VEC_FORMAT if STATE_WIDTHS[x] == 1])
    const VECTOR_FUNCTION_LIST = SVector{NUM_VECTOR_FUNCTIONS}([STATE_FUNCTIONS[x] for x in INPUT_VEC_FORMAT if STATE_WIDTHS[x] != 1])
    const SCALAR_FUNCTION_OFFSETS = SVector{NUM_SCALAR_FUNCTIONS}([sum([STATE_WIDTHS[x] for x in INPUT_VEC_FORMAT[1:i]]) + 1 for i in 0:(length(INPUT_VEC_FORMAT)-1) if STATE_WIDTHS[INPUT_VEC_FORMAT[i+1]] == 1])
    const VECTOR_FUNCTION_OFFSETS = SVector{NUM_VECTOR_FUNCTIONS}([sum([STATE_WIDTHS[x] for x in INPUT_VEC_FORMAT[1:i]]) for i in 0:(length(INPUT_VEC_FORMAT)-1) if STATE_WIDTHS[INPUT_VEC_FORMAT[i+1]] != 1])
end

begin
    Reinforce.finished(env::StrongholdEnvironment, s′) = current(env).piece == 11
    Reinforce.actions(env::StrongholdEnvironment, s) = 0:5
    Reinforce.state(env::StrongholdEnvironment) = env.state
    train_ind = 1
    function Reinforce.reset!(env::StrongholdEnvironment)
        env.resettable || return env
        env.stronghold = strongholds[rand(train)]
        env.room = length(env.stronghold)
        env.last_piece = 14
        env.last_exit = 0
        env.entry = 0
        env.reward = 0
        env.steps = 0
        env.state = get_state(env)
        return env
    end
    function Reinforce.step!(env::StrongholdEnvironment, s, a)
        env.steps += 1
        c = current(env)
        if a > piece_to_num_exits[c.piece]
            env.reward = INVALID_EXIT
        elseif a == 0 ? c.parent == 0 : c.exits[a] == 0
            env.reward = CLOSED_EXIT
        else
            env.reward = c.correctDirection == a ? CORRECT_DIRECTION : WRONG_DIRECTION
            env.last_piece = c.piece
            env.last_exit = a
            if a == 0
                env.entry = findfirst(==(env.room), env.stronghold[c.parent].exits)
                env.room = c.parent
            else
                env.room = c.exits[a]
                env.entry = 0
            end
        end
        env.state = get_state(env)
        c = current(env)
        if current(env).piece == 11
            env.reward = PORTAL_ROOM
        elseif c.piece > 9
            env.reward = min(env.reward, STUPID_ROOM)
        end
        return reward(env), state(env)
    end
    Reinforce.ismdp(::StrongholdEnvironment) = false
    Reinforce.maxsteps(::StrongholdEnvironment) = MAX_STEPS
end

begin
    struct FirstExitPolicy <: AbstractPolicy end
    Reinforce.action(π::FirstExitPolicy, r, s, A) = 1    
end

struct StrongholdReplayBuffer
    states::SizedVector{MAX_STEPS + 1, SVector{STATE_WIDTH, Float32}}
    actions::SVector{MAX_STEPS, Int8}
    rewards::SVector{MAX_STEPS, Int8}
end

function Base.iterate(rb::StrongholdReplayBuffer, i=1)
    (i > MAX_STEPS || rb.actions[i] == -1) && return nothing
    return (rb.states[i], rb.actions[i], rb.rewards[i], rb.states[i+1]), i + 1
end

return
