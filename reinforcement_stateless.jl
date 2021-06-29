begin
    const INPUT_VEC_ORDER = "DEPTH,DIRECTION,ENTRY,PREV_EXIT_INDEX_COMPAT,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5" # DEPTH,
    @time include("reinforcement_environment.jl")
end

@time begin
    println("Loading packages...")
    using Reinforce
    using Flux
    using BSON
    using Dates
    using ProgressMeter: @showprogress
    using BenchmarkTools: @benchmark
end

begin
    struct EpsilonGreedyPolicy{T} <: AbstractPolicy where T <: AbstractPolicy
        ϵ
        p::T
    end    
    Reinforce.action(π::EpsilonGreedyPolicy, r, s, A) = rand() < π.ϵ ? rand(A) : action(π.p, r, s, A)
end

begin
    struct QApproximatorPolicy <: AbstractPolicy
        approximator
        γ
    end
    Reinforce.action(π::QApproximatorPolicy, r, s, A) = A[argmax(π.approximator(s))]
    loss(qa::QApproximatorPolicy, s, a, r, s′) = (r + qa.γ*maximum(qa.approximator(s′)) - qa.approximator(s)[a])^2
end

begin
    struct NoBacktrackingPolicy <: AbstractPolicy
        p::QApproximatorPolicy
    end
    Reinforce.action(π::NoBacktrackingPolicy, r, s, A) = argmax(π.p.approximator(s)[2:6])
end

begin
    model = Chain(Dense(STATE_WIDTH, 20, relu), Dense(20, 8, relu), Dense(8, 6))
    qa = QApproximatorPolicy(model, 0.9)
    nb = NoBacktrackingPolicy(qa)
    eg = EpsilonGreedyPolicy(0.5, qa)
    opt = ADAM()
    train_ind = 0
    env = StrongholdEnvironment(strongholds[train[1]])
end

begin
    @showprogress 1 for i in 1:9000
        reset!(env)
        ep = Episode(env, eg)
        for (s, a, r, s′) in ep
            grads = Flux.gradient(Flux.params(model)) do 
                loss(qa, s, a + 1, r, s′)
            end
            
            Flux.Optimise.update!(opt, Flux.params(model), grads)
        end
        if i % 1000 == 0
            println(i, "\t", winrate(nb, dev))
            bson("models/tmp_rl_stateless_$i.bson")
        end
    end
end

begin
    env = StrongholdEnvironment(strongholds[dev[1]])
    ep = Episode(env, eg)
    for _ in ep end
    println(ep.total_reward)
end

function winrate(policy, dataset)
    wins = 0
    @showprogress 1 for i in dataset
        env = StrongholdEnvironment(strongholds[i])
        ep = Episode(env, policy)
        for _ in ep
        end
        if finished(env, 0)
            wins += 1
        end
    end
    return wins / length(dataset)
end

bson("models/stateless_420.bson", stateless_model = model)

begin
    env = StrongholdEnvironment(strongholds[dev[1]])
    s = state(env)
    println(argmax(model(s)), "\t", model(s))
end

begin
    r, s = step!(env, s, argmax(model(s)[2:6]))
    @show r, s
    println(argmax(model(s)[2:6]) + 1, "\t", model(s))
end