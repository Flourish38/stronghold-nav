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
    using Statistics
    using ProgressMeter: @showprogress
    using BenchmarkTools: @benchmark
end

begin
    struct EpsilonGreedyPolicy{T} <: AbstractPolicy where T <: AbstractPolicy
        ϵ
        p::T
    end
    Reinforce.reset!(eg::EpsilonGreedyPolicy) = reset!(eg.p)
    Reinforce.action(π::EpsilonGreedyPolicy, r, s, A) = rand() < π.ϵ ? rand(A) : action(π.p, r, s, A)
end

begin
    struct QApproximatorPolicy <: AbstractPolicy
        approximator
        γ
    end
    Reinforce.action(qa::QApproximatorPolicy, r, s, A) = A[argmax(qa.approximator(s))]
    Reinforce.reset!(qa::QApproximatorPolicy) = reset!(qa.approximator)
    function loss(qa::QApproximatorPolicy, s, a, r, s′) 
        reset!(qa)
        target = r + qa.γ*maximum(qa.approximator(s′))
        reset!(qa)
        prediction = qa.approximator(s)[a]
        (target - prediction)^2
    end
end

begin
    struct NoBacktrackingPolicy <: AbstractPolicy
        p::QApproximatorPolicy
    end
    Reinforce.reset!(nb::NoBacktrackingPolicy) = reset!(nb.p)
    Reinforce.action(π::NoBacktrackingPolicy, r, s, A) = argmax(π.p.approximator(s)[2:6])
end

begin
    struct RecurrentApproximator
        approximator
    end
    Reinforce.reset!(ra::RecurrentApproximator) = Flux.reset!(ra.approximator)
    (ra::RecurrentApproximator)(s) = ra.approximator.(s)[end]
end

function winrate(policy, dataset)
    wins = 0
    @showprogress 10 for i in dataset
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

begin
    model = Chain(LSTM(STATE_WIDTH, 24), LSTM(24, 8), Dense(8, 6))
    qa = QApproximatorPolicy(RecurrentApproximator(model), 0.95)
    nb = NoBacktrackingPolicy(qa)
    eg = EpsilonGreedyPolicy(0.2, qa)
    opt = ADAM()
    train_ind = 0
    env = StrongholdEnvironment(strongholds[train[1]])
end

begin
    @showprogress 1 for i in 1:3000
        ep = Episode(env, eg)
        for (s, a, r, s′) in ep
            grads = Flux.gradient(Flux.params(model)) do 
                loss(qa, s, a + 1, r, s′)
            end
            
            Flux.Optimise.update!(opt, Flux.params(model), grads)
        end
        if i % 100 == 0
            println(ep.total_reward)
            println(i, "\t", winrate(qa, dev[1:100]))
            println(i, "\t", winrate(nb, dev[1:100]))
        end
    end
end

begin
    println(winrate(qa, dev))
    println(winrate(nb, dev))
    #bson("models/tmp_rl_lstm_$i.bson")
end

begin
    Flux.reset!(model)
    env = StrongholdEnvironment(strongholds[dev[1]])
    s = state(env)
    output = model.(s)[end]
    println(argmax(output) - 1, "\t", output)
end

begin
    r, s = step!(env, s, argmax(output) - 1)
    @show r, s[end]
    Flux.reset!(model)
    output = model.(s)[end]
    println(argmax(output) - 1, "\t", output)
end