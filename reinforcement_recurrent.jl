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
    using Plots
    # Hack-y compatibility fix, I think unnecessary
    #=
    function (m::Flux.LSTMCell{A,V,<:Tuple{AbstractMatrix{T}, Any}})(s, x::SVector{N, T2}) where {A, V, T, N, T2<:Integer}
        return m(s, SVector{N, T}(x))
    end
    =#
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
    Reinforce.reset!(qa::QApproximatorPolicy) = Flux.reset!(qa.approximator)
    function loss(qa::QApproximatorPolicy, s, a, r, s′) 
        target = r + qa.γ*maximum(qa.target(s′))
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
    struct MultiThreadPolicy{P<:AbstractPolicy} <: AbstractPolicy
        ps::Vector{P}
        function MultiThreadPolicy(p::P) where P<:AbstractPolicy
            new{P}(vcat([p], [deepcopy(p) for _ in 2:Threads.nthreads()]))
        end
    end
    policy(mt::MultiThreadPolicy) = mt.ps[Threads.threadid()]
    Reinforce.reset!(mt::MultiThreadPolicy) = reset!(policy(mt))
    Reinforce.action(mt::MultiThreadPolicy, r, s, A) = action(policy(mt), r, s, A)
end

begin
    struct MultiThreadEnvironment{E<:AbstractEnvironment} <: AbstractEnvironment
        envs::Vector{E}
        function MultiThreadEnvironment(env::E) where E<:AbstractEnvironment
            new{E}([deepcopy(env) for _ in 1:Threads.nthreads()])
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
end

#=
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
=#

function winrate(policy, dataset)
    wins = 0
    #=@showprogress 10=# for i in dataset
        env = StrongholdEnvironment(strongholds[i], false)
        ep = Episode(env, policy)
        for _ in ep
        end
        if finished(env, 0)
            wins += 1
        end
    end
    return wins / length(dataset)
end

function winrate(policy::MultiThreadPolicy, dataset)
    wins = Threads.Atomic{Int}(0)
    Threads.@threads for i in dataset
        env = StrongholdEnvironment(strongholds[i], false)
        ep = Episode(env, policy)
        for _ in ep
        end
        if finished(env, 0)
            wins[] += 1
        end
    end
    return wins[] / length(dataset)
end

begin  # non-recurrent
    model = Chain(Dense(STATE_WIDTH, 64), Dense(64, 64), Dense(64, 6))
    #approx = DeRecurrentApproximator(model)
end

begin  # recurrent
    model = Chain(LSTM(STATE_WIDTH, 64), LSTM(64, 64), Dense(64, 6))
    #approx = DeRecurrentApproximator(model)
end

begin  # load model
    loaded_bson = BSON.load("models/rl_rnn_1.bson")
    model = loaded_bson[:rl_model]
end

begin
    qa = QApproximatorPolicy(model, deepcopy(model), 0.95)
    nb = NoBacktrackingPolicy(qa)
    env = StrongholdEnvironment()
    opt = ADAM()
    iters = 0
end

begin
    opt = loaded_bson[:adam]
    iters = loaded_bson[:iters]
end

function update_buffer!(buffer::CircularVectorBuffer{StrongholdReplayBuffer}, env, policy, M)
    env.resettable = true
    for _ in 1:M
        ep = Episode(env, policy)
        states = SizedVector{MAX_STEPS + 1, SVector{STATE_WIDTH, Int8}}(undef)
        states[1] = state(env)
        actions = @MVector fill(Int8(-1), MAX_STEPS)
        rewards = @MVector fill(Int8(-1), MAX_STEPS)
        for (s, a, r, s′) in ep
            actions[env.steps] = a
            rewards[env.steps] = r
            states[env.steps + 1] = s′
        end
        push!(buffer, StrongholdReplayBuffer(states, actions, rewards))
    end
end

function update_buffer!(buffer::CircularVectorBuffer{StrongholdReplayBuffer}, _, policy::MultiThreadPolicy, M)
    temp = SizedVector{M, StrongholdReplayBuffer}(undef)
    Threads.@threads for i in 1:M
        env = StrongholdEnvironment()
        ep = Episode(env, policy)
        states = SizedVector{MAX_STEPS + 1, SVector{STATE_WIDTH, Int8}}(undef)
        states[1] = state(env)
        actions = @MVector fill(Int8(-1), MAX_STEPS)
        rewards = @MVector fill(Int8(-1), MAX_STEPS)
        for (s, a, r, s′) in ep
            actions[env.steps] = a
            rewards[env.steps] = r
            states[env.steps + 1] = s′
        end
        temp[i] = StrongholdReplayBuffer(states, actions, rewards)
    end
    for srb in temp
        push!(buffer, srb)
    end
end

begin
    buffer_size = 10000
    replay_buffer = CircularVectorBuffer{StrongholdReplayBuffer}(buffer_size)
    rp = RandomPolicy()
    best_dev_winrate = @time winrate(MultiThreadPolicy(qa), dev)
    eg = EpsilonGreedyPolicy(0.02, qa)
    #update_buffer!(replay_buffer, env, rp, buffer_size)
    @time update_buffer!(replay_buffer, env, MultiThreadPolicy(eg), buffer_size)
end

function mean_loss_replay(replay, qa)
    n = 0
    total = 0
    Flux.reset!(qa.approximator)
    Flux.reset!(qa.target)
    for (s, a, r, s′) in replay
        n += 1
        if n == 1
            qa.target(s)
        end
        total += loss(qa, s, a + 1, r, s′)
    end
    return total / n
end

function replay_batch_loss(batch, qa)
    return sum(mean_loss_replay(replay, qa) for replay in batch)
end

function train_step!(params, opt, batch, qa)
    grads = Flux.gradient(params) do 
        replay_batch_loss(batch, qa)
    end
    Flux.Optimise.update!(opt, params, grads)
end

function train_step_mt!(params, opt, batch, qa)
    mt = MultiThreadPolicy(qa)
    temp = Vector{Flux.Zygote.Grads}(undef, Threads.nthreads())
    k = length(batch)/Threads.nthreads()

    Threads.@threads for i in 0:k:length(batch)-1
        p = policy(mt)
        sub_batch = batch[ceil(Int, 1+i):ceil(Int, k+i)]
        temp[Threads.threadid()] = Flux.gradient(Flux.params(p.approximator)) do 
            replay_batch_loss(sub_batch, p)
        end
    end
    
    grads = temp[1]
    vs = collect.(values.(temp))
    for v in vs[2:end]
        vs[1] .+= v
    end

    Flux.Optimise.update!(opt, params, grads)
end

@time begin
    mt = MultiThreadPolicy(qa)
    temp = Vector{Flux.Zygote.Grads}(undef, 4)
    k = length(batch)/Threads.nthreads()
    Threads. @threads for i in 0:k:length(batch)-1
        p = policy(mt)
        sub_batch = batch[ceil(Int, 1+i):ceil(Int, k+i)]
        temp[Threads.threadid()] = Flux.gradient(Flux.params(p.approximator)) do 
            replay_batch_loss(sub_batch, p)
        end
    end
end

function training_loop(M, K, B, update_interval, qa::QApproximatorPolicy, eg::EpsilonGreedyPolicy{QApproximatorPolicy}, model, opt, replay_buffer::CircularVectorBuffer{StrongholdReplayBuffer})
    global best_dev_winrate, dev, iters
    next_time = now() + update_interval
    while true
        iters += 1
        qa.target = deepcopy(qa.approximator)
        update_buffer!(replay_buffer, 0, MultiThreadPolicy(eg), M)
        
        for i = 1:K
            batch = rand(replay_buffer, B)
            train_step_mt!(Flux.params(model), opt, batch, qa)
            GC.gc(false)
        end
        
        if now() > next_time
            println(iters, "\t", now())
            next_time = now() + update_interval
        end
        
        mt = MultiThreadPolicy(qa)
        dev_subset = rand(dev, 100)
        subset_winrate = winrate(mt, dev_subset)
        if subset_winrate >= best_dev_winrate
            w = winrate(mt, dev)
            println(iters, "\t", subset_winrate, "\t", w)
            if w > best_dev_winrate
                println("New best! +", w - best_dev_winrate)
                best_dev_winrate = w
                bson("models/tmp/rl_stateless_a_$iters.bson", rl_model = model, adam = opt, iters = iters)
            end
        end
    end
end

begin
    update_interval = Minute(10)
    M = 100
    B = Threads.nthreads() * 1
    K = M ÷ B
    Reinforce.maxsteps(env::StrongholdEnvironment) = MAX_STEPS
    training_loop(M, K, B, update_interval, qa, eg, model, opt, replay_buffer)
end

begin  # When you interrupt it and it would have maybe saved, do this
    w = winrate(qa, dev)
    println(iters, "\t", w)
    if w > best_dev_winrate
        println("New best! +", w - best_dev_winrate)
        best_dev_winrate = w
        bson("models/tmp/rl_recurrent_d_$iters.bson", rl_model = model, adam = opt, iters = iters)
    end
end

@time begin
    println(winrate(qa, dev))
    println(winrate(nb, dev))
    println(winrate(eg, dev))
end

function win_distribution(policy, dataset)
    distribution = Int[]
    max_steps_win = 0
    @showprogress 1 for i in dataset
        env = StrongholdEnvironment(strongholds[i], false)
        ep = Episode(env, policy)
        for _ in ep
        end
        if finished(env, 0)
            push!(distribution, env.steps)
            if env.steps > max_steps_win
                #println(i, "\t", env.steps)
                max_steps_win = env.steps
            end
        end
    end
    return distribution
end

begin
    Reinforce.maxsteps(env::StrongholdEnvironment) = 80
    distr = win_distribution(qa, dev)
end

begin
    r = 1:80#maxsteps(env)
    y_graph = [sum(distr .== x) for x in r]
    Plots.bar(r, y_graph)
    xlabel!("# Steps to find portal room")
    ylabel!("Count")
    Plots.savefig("win_distribution.png")
end

begin
    y_graph_cumulative = [sum(y_graph[1:x]) for x in r] ./ length(dev)
    Plots.bar(r, y_graph_cumulative)
    xlabel!("# Steps to find portal room")
    ylabel!("Success rate (cumulative)")
    Plots.savefig("win_cumulative.png")
end

begin
    best_model = BSON.load("models/rl_rnn_1.bson")[:rl_model]
    #mhm = Chain(LSTM(STATE_WIDTH, 64), LSTM(64, 64), Dense(64, 6))
    #Flux.loadparams!(mhm, Flux.params(best_model))
    #best_model = mhm
    bqa = QApproximatorPolicy(best_model, best_model, 0.95)
    bnb = NoBacktrackingPolicy(bqa)
    beg = EpsilonGreedyPolicy(0.1, bqa)
    Reinforce.maxsteps(env::StrongholdEnvironment) = MAX_STEPS
end

begin
    println(winrate(bqa, dev))
    println(winrate(bnb, dev))    
    println(winrate(beg, dev))
end

begin
    Flux.reset!(model)
    env = StrongholdEnvironment(strongholds[dev[10]], false)
    s = state(env)
    output = model(s)
    println(argmax(output) - 1, "\t", output)
end

begin
    r, s = step!(env, s, argmax(output) - 1)
    @show r, s
    output = model(s)
    println(argmax(output) - 1, "\t", output)
end