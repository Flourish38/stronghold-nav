include("reinforcement_environment.jl")

print_room(env::StrongholdEnvironment) = print_room(env.stronghold, env.room)
print_room(stronghold, r) = print_room(stronghold, stronghold[r])
function print_room(stronghold, r::Room)
    println(
        _pieces_vec[r.piece], "\r\t\t  ",
        r.depth, "\t",
        _pieces_vec[get_piece(stronghold, r.parent)], "\r\t\t\t\t\t  ",
        [_pieces_vec[get_piece(stronghold, r.exits[i])] for i in 1:PIECE_TO_NUM_EXITS[r.piece]]
    )
end

function print_correct_path(stronghold)
    env = StrongholdEnvironment(stronghold, false)
    p = OraclePolicy(env)
    print_room(env)
    for (s, a, r, s′) in Episode(env, p)
        println(a, "\t", r)
        print_room(env)
    end
    return env.replay
end

function random_replay()
    env = StrongholdEnvironment(rand(strongholds), false)
    p = RandomPolicy()
    for _ in Episode(env, p); end
    return env.replay
end

struct FirstExitPolicy <: AbstractPolicy end
Reinforce.action(π::FirstExitPolicy, r, s, A) = 1    

# Oracle policies need to have the environment to know which way to go, so need to be treated differently by functions.
abstract type AbstractOraclePolicy <: AbstractPolicy end
mutable struct OraclePolicy <: AbstractOraclePolicy env::Union{StrongholdEnvironment, Nothing} end
Reinforce.action(π::OraclePolicy, r, s, A) = current(π.env).targetDirection

struct EpsilonGreedyPolicy{T} <: AbstractPolicy where T <: AbstractPolicy
    ϵ
    p::T
end
Reinforce.reset!(eg::EpsilonGreedyPolicy) = reset!(eg.p)
Reinforce.action(π::EpsilonGreedyPolicy, r, s, A) = rand() < π.ϵ ? rand(A) : action(π.p, r, s, A)

struct MultiThreadPolicy{P<:AbstractPolicy} <: AbstractPolicy
    ps::Vector{P}
    function MultiThreadPolicy(p::P) where P<:AbstractPolicy
        new{P}(vcat([p], [deepcopy(p) for _ in 2:Threads.nthreads()]))
    end
end
policy(mt::MultiThreadPolicy) = mt.ps[Threads.threadid()]
Reinforce.reset!(mt::MultiThreadPolicy) = reset!(policy(mt))
Reinforce.action(mt::MultiThreadPolicy, r, s, A) = action(policy(mt), r, s, A)


struct MultiThreadEnvironment{E<:AbstractEnvironment} <: AbstractEnvironment
    envs::Vector{E}
    function MultiThreadEnvironment(env::E) where E<:AbstractEnvironment
        new{E}(vcat([env], [deepcopy(env) for _ in 2:Threads.nthreads()]))
    end
end
environment(env::MultiThreadEnvironment) = env.envs[Threads.threadid()]
Reinforce.reset!(env::MultiThreadEnvironment) = reset!(environment(env))
Reinforce.actions(env::MultiThreadEnvironment, s) = actions(environment(env), s)
Reinforce.step!(env::MultiThreadEnvironment, s, a) = step!(environment(env), s, a)
Reinforce.finished(env::MultiThreadEnvironment, s′) = finished(environment(env), s′)
Reinforce.state(env::MultiThreadEnvironment) = state(environment(env))
Reinforce.reward(env::MultiThreadEnvironment) = reward(environment(env))
Reinforce.ismdp(env::MultiThreadEnvironment) = ismdp(environment(env))
Reinforce.maxsteps(env::MultiThreadEnvironment) = maxsteps(environment(env))

# Necessary for wrapper policies like NoBacktrackingPolicy that rely on the model output directly.
abstract type AbstractModelPolicy <: AbstractPolicy end

struct NoBacktrackingPolicy{P <: AbstractModelPolicy} <: AbstractPolicy
    p::P
end
Reinforce.reset!(nb::NoBacktrackingPolicy) = reset!(nb.p)
Reinforce.action(π::NoBacktrackingPolicy, r, s, A) = argmax(π.p.model(s)[2:6])

struct NeverMissTargetPolicy{P <: AbstractPolicy} <: AbstractPolicy
    p::P
    target_piece::Int8
    function NeverMissTargetPolicy(p::P, target_piece=11) where P <: AbstractPolicy
        new{P}(p, target_piece)
    end
end
Reinforce.reset!(π::NeverMissTargetPolicy) = reset!(π.p)
# Assumes that PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5 are all contiguous in s
function Reinforce.action(π::NeverMissTargetPolicy, r, s, A)
    j = VECTOR_FUNCTION_OFFSETS[findfirst(==("PARENT_ROOM"), INPUT_VEC_FORMAT) - NUM_SCALAR_FUNCTIONS] + π.target_piece
    for i in 0:5
        if s[j + 14i] == 1
            return i
        end
    end
    return action(π.p, r, s, A)
end

function winrate(policy, dataset)
    return winrate(MultiThreadPolicy(policy), dataset)
    #=
    wins = 0
    #=@showprogress 10=# for i in dataset
        env = StrongholdEnvironment(strongholds[i], false)
        if typeof(policy) <: AbstractOraclePolicy
            policy.env = env
        end
        ep = Episode(env, policy)
        for _ in ep
        end
        if finished(env, 0)
            wins += 1
        end
    end
    return wins / length(dataset)
    =#
end

# This is a significant speedup over single-threaded
function winrate(p::MultiThreadPolicy, dataset)
    wins = Threads.Atomic{Int}(0)
    Threads.@threads for i in dataset
        env = StrongholdEnvironment(strongholds[i], false)
        if typeof(policy(p)) <: AbstractOraclePolicy
            policy(p).env = env
        end
        ep = Episode(env, p)
        for _ in ep
        end
        if finished(env, 0)
            wins[] += 1
        end
    end
    return wins[] / length(dataset)
end

function generate_replays(p, M)
    return generate_replays(MultiThreadPolicy(p), M)
    #=
    output = SizedVector{M, StrongholdReplay}(undef)
    env = StrongholdEnvironment()  # resettable
    for i in 1:M
        reset!(env)
        reset!(p)
        for _ in Episode(env, p); end
        output[i] = env.replay
    end
    return output
    =#
end

function generate_replays(p::MultiThreadPolicy, M)
    output = SizedVector{M, StrongholdReplay}(undef)
    env = MultiThreadEnvironment(StrongholdEnvironment())  # resettable
    Threads.@threads for i in 1:M
        tl_env = environment(env)
        tl_p = policy(p)
        reset!(tl_env)
        reset!(tl_p)
        for _ in Episode(tl_env, tl_p); end
        output[i] = tl_env.replay
    end
    return output
end

return
