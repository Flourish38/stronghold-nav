using Reinforce

# Generic policies

struct EpsilonGreedyPolicy{T} <: AbstractPolicy where T <: AbstractPolicy
    ϵ
    p::T
end
Reinforce.reset!(eg::EpsilonGreedyPolicy) = reset!(eg.p)
Reinforce.action(π::EpsilonGreedyPolicy, r, s, A) = rand() < π.ϵ ? rand(A) : action(π.p, r, s, A)


abstract type AbstractOraclePolicy <: AbstractPolicy end


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

# StrongholdEnvironment-specific policies

struct FirstExitPolicy <: AbstractPolicy end
Reinforce.action(π::FirstExitPolicy, r, s, A) = 1


mutable struct OraclePolicy <: AbstractOraclePolicy env::Union{StrongholdEnvironment, Nothing} end
Reinforce.action(π::OraclePolicy, r, s, A) = current(π.env).correctDirection


mutable struct QApproximatorPolicy <: AbstractPolicy
    approximator
    target
    γ
end
Reinforce.action(qa::QApproximatorPolicy, r, s, A) = A[argmax(qa.approximator(s))]
Reinforce.reset!(qa::QApproximatorPolicy) = Flux.reset!(qa.approximator)
function loss(qa::QApproximatorPolicy, s, a, r, s′) 
    target = r + qa.γ*maximum(qa.target(s′))
    prediction = qa.approximator(s)[a]
    (target - prediction)^2
end


struct NoBacktrackingPolicy <: AbstractPolicy
    p::QApproximatorPolicy
end
Reinforce.reset!(nb::NoBacktrackingPolicy) = reset!(nb.p)
Reinforce.action(π::NoBacktrackingPolicy, r, s, A) = argmax(π.p.approximator(s)[2:6])


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

# Utility functions

function winrate(policy, dataset)
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

