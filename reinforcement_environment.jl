@time include("load_strongholds.jl")

@time begin
    println("Loading packages...")
    using BenchmarkTools: @benchmark
    using ProgressMeter
    using Statistics
    using Dates
    using BSON
    using Reinforce
    #using IntervalSets
    using Flux
    using Flux: onehot,onecold
end

mutable struct StrongholdEnvironment <: AbstractEnvironment
    reward::Int
    state::Vector{Int8}
    steps::Int
    stronghold::Vector{Room}
    room::Int16
    last_piece::Int8
    last_exit::Int8
    entry::Int8
    function StrongholdEnvironment(s::Vector{Room})
        env = new(0, [], 0, s, length(s), 14, 0, 0)
        env.state = get_state(env)
        env
    end
end

begin
    current(env::StrongholdEnvironment)::Room = env.stronghold[env.room]
    get_piece(env::StrongholdEnvironment, room)::Int8 = get_piece(env.stronghold, room)
    encode_piece(s, room) = onehot(get_piece(s, room), 1:14)
    get_state(env::StrongholdEnvironment) = reduce(vcat, x(env) for x in STATE_FUNCTION_LIST)
end


begin
    const PORTAL_ROOM = 10
    const CORRECT_DIRECTION = 1
    const WRONG_DIRECTION = -2
    const CLOSED_EXIT = -5
    const INVALID_EXIT = -10
    const INPUT_VEC_ORDER = split("DEPTH,DIRECTION,ENTRY,PREV_EXIT_INDEX_COMPAT,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5", ',')
    const STATE_FUNCTIONS = Dict(
        "CURRENT" => (env::StrongholdEnvironment) -> onehot(current(env).piece, 1:14),
        "PARENT_ROOM" => (env::StrongholdEnvironment) -> encode_piece(env, current(env).parent),
        "PREVIOUS_ROOM" => (env::StrongholdEnvironment) -> onehot(env.last_piece, 1:14),
        "EXIT_1" => (env::StrongholdEnvironment) -> encode_piece(env, current(env).exits[1]),
        "EXIT_2" => (env::StrongholdEnvironment) -> encode_piece(env, current(env).exits[2]),
        "EXIT_3" => (env::StrongholdEnvironment) -> encode_piece(env, current(env).exits[3]),
        "EXIT_4" => (env::StrongholdEnvironment) -> encode_piece(env, current(env).exits[4]),
        "EXIT_5" => (env::StrongholdEnvironment) -> encode_piece(env, current(env).exits[5]),
        "PREV_EXIT_INDEX" => (env::StrongholdEnvironment) -> SVector{1, Int8}(env.last_exit),
        "PREV_EXIT_INDEX_COMPAT" => (env::StrongholdEnvironment) -> onehot(env.last_exit, 0:5),
        "DIRECTION" => (env::StrongholdEnvironment) -> onehot(current(env).orientation, 1:4),
        "DEPTH" => (env::StrongholdEnvironment) -> SVector{1, Int8}(current(env).depth),
        "CONSTANT" => (env::StrongholdEnvironment) -> SVector{1, Int8}(0),
        "DOWNWARDS" => (env::StrongholdEnvironment) -> SVector{1, Int8}(env.entry == 0 ? 1 : 0),
        "ENTRY" => (env::StrongholdEnvironment) -> onehot(env.entry, 0:5)
    )
    const STATE_FUNCTION_LIST = [STATE_FUNCTIONS[x] for x in INPUT_VEC_ORDER]
    #= const STATE_SPACES = Dict(
        "CURRENT" => [0..1 for _ in 1:14],
        "PARENT_ROOM" => [0..1 for _ in 1:14],
        "PREVIOUS_ROOM" => [0..1 for _ in 1:14],
        "EXIT_1" => [0..1 for _ in 1:14],
        "EXIT_2" => [0..1 for _ in 1:14],
        "EXIT_3" => [0..1 for _ in 1:14],
        "EXIT_4" => [0..1 for _ in 1:14],
        "EXIT_5" => [0..1 for _ in 1:14],
        "PREV_EXIT_INDEX" => [0..5],
        "PREV_EXIT_INDEX_COMPAT" => [0..1 for _ in 0:5],
        "DIRECTION" => [0..1 for _ in 1:4],
        "DEPTH" => [0..typemax(Int8)],
        "CONSTANT" => [0..0],
        "DOWNWARDS" => [0..1],
        "ENTRY" => [0..1 for _ in 0:5]
    ) =#
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
        return StrongholdEnvironment(strongholds[train[train_ind]])
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
            env.state = get_state(env)
        end
        if current(env).piece == 11
            env.reward = PORTAL_ROOM
        end
        return env.reward, env.state
    end
    Reinforce.ismdp(::StrongholdEnvironment) = false
    Reinforce.maxsteps(::StrongholdEnvironment) = 0
end


@time begin
    train_ind = 0
    env = StrongholdEnvironment(strongholds[1])
    policy = RandomPolicy()
    wins = 0
    total_niter = 0
    @showprogress for _ in 1:90000
        reset!(env)
        ep = Episode(env, policy)
        for (s, a, r, s′) in ep
            #println(r)
        end
        total_niter += ep.niter
    end
    #println(wins)
    println(total_niter/90000)
    #println(ep.total_reward)
    #println(ep.niter)
end


