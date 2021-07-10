begin
    const INPUT_VEC_ORDER = "DEPTH,DIRECTION,ENTRY,PREV_EXIT_INDEX_COMPAT,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5" # DEPTH,
    @time include("reinforcement_environment.jl")
end

@time begin
    println("Loading packages...")
    using Reinforce
    using Flux
    using BSON
    using CircularArrayBuffers
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
    mutable struct QApproximatorPolicy <: AbstractPolicy
        approximator
        target
        γ
    end
    Reinforce.action(qa::QApproximatorPolicy, r, s, A) = A[argmax(qa.approximator(s))]
    Reinforce.reset!(qa::QApproximatorPolicy) = reset!(qa.approximator)
    function loss(qa::QApproximatorPolicy, s, a, r, s′) 
        reset!(qa.target)
        target = r + qa.γ*maximum(qa.target(s′))
        reset!(qa.approximator)
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

begin
    struct DeRecurrentApproximator
        approximator
    end
    Reinforce.reset!(ra::DeRecurrentApproximator) = nothing
    (ra::DeRecurrentApproximator)(s) = ra.approximator(s[end])
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

begin  # non-recurrent
    model = Chain(Dense(STATE_WIDTH, 24), Dense(24, 8), Dense(8, 6))
    approx = DeRecurrentApproximator(model)
end

begin  # recurrent
    model = Chain(LSTM(STATE_WIDTH, 24), LSTM(24, 8), Dense(8, 6))
    approx = DeRecurrentApproximator(model)
end

begin
    qa = QApproximatorPolicy(approx, deepcopy(approx), 0.95)
    nb = NoBacktrackingPolicy(qa)
    env = StrongholdEnvironment(strongholds[train[1]])
    opt = ADAM()
end

@time begin
    buffer_size = 4000
    replay_buffer = CircularVectorBuffer{StrongholdReplayBuffer}(buffer_size)
    rp = RandomPolicy()
    for i in 1:buffer_size
        ep = Episode(env, rp)
        actions = @MVector fill(Int8(-1), MAX_STEPS)
        rewards = @MVector fill(Int8(-1), MAX_STEPS)
        for (s, a, r, s′) in ep
            actions[env.steps] = a
            rewards[env.steps] = r
        end
        push!(replay_buffer, StrongholdReplayBuffer(env, actions, rewards))
    end
end



function mean_loss_replay(replay, qa)
    return mean(loss(qa, s, a + 1, r, s′) for (s, a, r, s′) in replay)
end

function replay_batch_loss(batch, qa)
    return sum(mean_loss_replay(replay, qa) for replay in batch)
end

function train_step!(params, opt, batch, qa)
    grads = Flux.gradient(params) do 
        replay_batch_loss(batch, qa)
    end
    Flux.Optimise.update!(opt, Flux.params(model), grads)
end

function update_buffer!(buffer, env, policy, M)
    for _ in 1:M
        ep = Episode(env, policy)
        actions = @MVector fill(Int8(-1), MAX_STEPS)
        rewards = @MVector fill(Int8(-1), MAX_STEPS)
        for (s, a, r, s′) in ep
            actions[env.steps] = a
            rewards[env.steps] = r
        end
        push!(buffer, StrongholdReplayBuffer(env, actions, rewards))
    end
end

begin
    M = 100
    K = 25
    B = 20
    iters = 0
    eg = EpsilonGreedyPolicy(0.3, qa)
    while true
        iters += 1
        qa.target = deepcopy(qa.approximator)
        update_buffer!(replay_buffer, env, eg, M)
        
        for i = 1:K
            batch = rand(replay_buffer, B)
            train_step!(Flux.params(model), opt, batch, qa)
        end

        dev_subset = rand(dev, 100)
        println(iters, "\t", winrate(qa, dev_subset), "\t", winrate(nb, dev_subset), "\t", now())
    end
end

@time begin
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