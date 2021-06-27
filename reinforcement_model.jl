@time include("load_strongholds.jl")

@time begin
    println("Loading packages...")
    using BenchmarkTools: @benchmark
    using Statistics
    using Dates
    using BSON
    using ReinforcementLearning
    import ReinforcementLearning: is_discrete_space
    using IntervalSets
    using Flux
    using Flux: onehot,onecold,InvDecay
end

mutable struct StrongholdEnv <: AbstractEnv
    reward::Int
    steps::Int
    stronghold::Vector{Room}
    room::Int16
    last_piece::Int8
    last_exit::Int8
    entry::Int8
    function StrongholdEnv(s::Vector{Room})
        new(0, 0, s, length(s), 14, 0, 0)
    end
end

begin
    current(env::StrongholdEnv) = env.stronghold[env.room]
    get_piece(env::StrongholdEnv, room) = get_piece(env.stronghold, room)
    encode_piece(s, room) = onehot(get_piece(s, room), 1:14)
end


begin
    const PORTAL_ROOM = 10
    const CORRECT_DIRECTION = 1
    const WRONG_DIRECTION = -2
    const CLOSED_EXIT = -5
    const INVALID_EXIT = -10
    const INPUT_VEC_ORDER = split("DEPTH,DIRECTION,ENTRY,PREV_EXIT_INDEX_COMPAT,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5", ',')
    const STATE_FUNCTIONS = Dict(
        "CURRENT" => (env::StrongholdEnv) -> onehot(current(env).piece, 1:14),
        "PARENT_ROOM" => (env::StrongholdEnv) -> encode_piece(env, current(env).parent),
        "PREVIOUS_ROOM" => (env::StrongholdEnv) -> onehot(env.last_piece, 1:14),
        "EXIT_1" => (env::StrongholdEnv) -> encode_piece(env, current(env).exits[1]),
        "EXIT_2" => (env::StrongholdEnv) -> encode_piece(env, current(env).exits[2]),
        "EXIT_3" => (env::StrongholdEnv) -> encode_piece(env, current(env).exits[3]),
        "EXIT_4" => (env::StrongholdEnv) -> encode_piece(env, current(env).exits[4]),
        "EXIT_5" => (env::StrongholdEnv) -> encode_piece(env, current(env).exits[5]),
        "PREV_EXIT_INDEX" => (env::StrongholdEnv) -> [env.last_exit],
        "PREV_EXIT_INDEX_COMPAT" => (env::StrongholdEnv) -> onehot(env.last_exit, 0:5),
        "DIRECTION" => (env::StrongholdEnv) -> onehot(current(env).orientation, 1:4),
        "DEPTH" => (env::StrongholdEnv) -> [current(env).depth],
        "CONSTANT" => (env::StrongholdEnv) -> [0],
        "DOWNWARDS" => (env::StrongholdEnv) -> [env.entry == 0 ? 1 : 0],
        "ENTRY" => (env::StrongholdEnv) -> onehot(env.entry, 0:5)
    )
    const STATE_SPACES = Dict(
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
    )
end

begin
    RLBase.is_terminated(env::StrongholdEnv) = current(env).piece == 11 || env.steps >= 1000
    RLBase.reward(env::StrongholdEnv) = env.reward
    RLBase.state(env::StrongholdEnv) = reduce(vcat, [STATE_FUNCTIONS[x](env) for x in INPUT_VEC_ORDER])
    RLBase.state_space(env::StrongholdEnv) = Space(reduce(vcat, [STATE_SPACES[x] for x in INPUT_VEC_ORDER]))
    RLBase.action_space(env::StrongholdEnv) = 0:5
    train_ind = 1
    function RLBase.reset!(env::StrongholdEnv)
        global train_ind
        train_ind += 1
        if train_ind > length(train)
            train_ind = 1
        end
        env.stronghold = strongholds[train[train_ind]]
        env.room = length(env.stronghold)
        env.steps = 0
        env.reward = 0
        env.last_piece = 14
        env.last_exit = 0
        env.entry = 0
    end
    function (env::StrongholdEnv)(action)
        env.steps += 1
        c = current(env)
        if action > piece_to_num_exits[c.piece]
            env.reward = INVALID_EXIT
        elseif action == 0 ? c.parent == 0 : c.exits[action] == 0
            env.reward = CLOSED_EXIT
        else
            env.reward = c.correctDirection == action ? CORRECT_DIRECTION : WRONG_DIRECTION
            env.last_piece = c.piece
            env.last_exit = action
            if action == 0
                env.entry = findfirst(==(env.room), env.stronghold[c.parent].exits)
                env.room = c.parent
            else
                env.room = c.exits[action]
                env.entry = 0
            end
        end
        if current(env).piece == 11
            env.reward = PORTAL_ROOM
        end
    end

    # These are all default I am just messing around with them
    NumAgentStyle(::StrongholdEnv) = SINGLE_AGENT
    DynamicStyle(::StrongholdEnv) = SEQUENTIAL
    ActionStyle(::StrongholdEnv) = MINIMAL_ACTION_SET
    InformationStyle(::StrongholdEnv) = IMPERFECT_INFORMATION
    #StateStyle(::StrongholdEnv) = Observation{Any}()
    RewardStyle(::StrongholdEnv) = STEP_REWARD
    #UtilityStyle(::StrongholdEnv) = GENERAL_SUM
    ChanceStyle(::StrongholdEnv) = STOCHASTIC
end

begin
    # Fixing their broken code...
    is_discrete_space(::Type{<:AbstractVector}) = true
    is_discrete_space(::Type{<:Tuple}) = true
    is_discrete_space(::Type{<:NamedTuple}) = true
end

begin # This works!
    train_ind = 1
    @time run(RandomPolicy(), StrongholdEnv(strongholds[train[train_ind]]), StopAfterEpisode(90000), TotalRewardPerEpisode())
end

begin
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    Chain(Dense(115, 20, relu), Dense(20, 8, relu), Dense(8, 6)), ADAM()
                )
            ),
        explorer = EpsilonGreedyExplorer(0.1)
    )
    train_ind = 1
    env = StrongholdEnv(strongholds[train[train_ind]])
    env = discrete2standard_discrete(env)
    #agent = Agent(policy, CircularVectorSARTSATrajectory(state = Vector{Int8}, capacity = 128))
    @time run(policy, env, StopAfterEpisode(100), TotalRewardPerEpisode())
end

