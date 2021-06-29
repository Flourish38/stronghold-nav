println(INPUT_VEC_ORDER)

include("load_strongholds.jl")

@time begin
    println("Loading packages...")
    using StaticArrays
    using Reinforce
end

mutable struct StrongholdEnvironment <: AbstractEnvironment
    reward::Float64
    state::Vector{Int8}
    stronghold::Vector{Room}
    room::Int16
    last_piece::Int8
    last_exit::Int8
    entry::Int8
    function StrongholdEnvironment(s::Vector{Room})
        env = new(0, [], s, length(s), 14, 0, 0)
        env.state = get_state(env)
        env
    end
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
    const PORTAL_ROOM = 50
    const CORRECT_DIRECTION = 1
    const WRONG_DIRECTION = -2
    const CLOSED_EXIT = -5
    const INVALID_EXIT = -10
    const INPUT_VEC_FORMAT = split(INPUT_VEC_ORDER, ',')
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
    train_ind = 1
    function Reinforce.reset!(env::StrongholdEnvironment)
        global train_ind
        train_ind += 1
        if train_ind > length(train)
            train_ind = 1
        end
        a = strongholds[train[train_ind]]
        env.room = length(env.stronghold)
        env.last_piece = 14
        env.last_exit = 0
        env.entry = 0
        env.reward = 0
        env.state = get_state(env)
        return env
    end
    function Reinforce.step!(env::StrongholdEnvironment, s, a)
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
            env.state = get_state(env)
        end
        if current(env).piece == 11
            env.reward = PORTAL_ROOM
        end
        return env.reward, env.state
    end
    Reinforce.ismdp(::StrongholdEnvironment) = false
    Reinforce.maxsteps(::StrongholdEnvironment) = 100
end

begin
    struct FirstExitPolicy <: AbstractPolicy end
    Reinforce.action(π::FirstExitPolicy, r, s, A) = 1    
end

return
