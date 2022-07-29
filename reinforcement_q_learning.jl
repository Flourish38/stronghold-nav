@time begin
    const INPUT_VEC_ORDER = "DEPTH,DIRECTION,ENTRY,PREV_EXIT_INDEX_COMPAT,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5" # DEPTH,
    include("reinforcement_environment.jl")
    include("utils.jl")
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
    using UnicodePlots
end

begin
    # The OccasionalOraclePolicy will randomly turn into an oracle policy if a random number is > `chance`. It is of dubious utility.
    mutable struct OccasionalOraclePolicy{P<:AbstractPolicy} <: AbstractOraclePolicy
        state::Int
        chance::Float64
        p::P
        env::Union{Nothing, StrongholdEnvironment}
        function OccasionalOraclePolicy(chance, p::P) where P<:AbstractPolicy
            new{P}(0, chance, p, nothing)
        end
    end
    function Reinforce.reset!(oo::OccasionalOraclePolicy) 
        oo.state = 0
        reset!(oo.p)
    end
    function Reinforce.action(oo::OccasionalOraclePolicy, r, s, A)
        if oo.state == 1
            return current(oo.env).correctDirection
        elseif rand() * (oo.env.steps + 1) < oo.chance
            oo.state = 1
            return action(oo, r, s, A)
        else
            return action(oo.p, r, s, A)
        end
    end
end

begin  # non-recurrent
    model = Chain(Dense(STATE_WIDTH, 64), Dense(64, 64), Dense(64, 6))
end

begin  # recurrent
    model = Chain(LSTM(STATE_WIDTH, 64), LSTM(64, 64), Dense(64, 6))
end

begin  # load model
    loaded_bson = BSON.load("models/tmp/rl_rnn_a2_434.bson")
    model = loaded_bson[:rl_model]
end

begin
    qa = QApproximatorPolicy(model, deepcopy(model), 0.95)
    nb = NoBacktrackingPolicy(qa)
    eg = EpsilonGreedyPolicy(0.3, qa)
    env = StrongholdEnvironment()
end

begin
    opt = ADAM()
    iters = 0
end

begin
    opt = loaded_bson[:adam]
    iters = loaded_bson[:iters]
end

struct StrongholdReplayBuffer
    states::SizedVector{MAX_STEPS + 1, SVector{STATE_WIDTH, Float32}}  # Float32 to avoid converting to Float32 multiple times later
    actions::SVector{MAX_STEPS, Int8}
    rewards::SVector{MAX_STEPS, Int8}
end

function Base.iterate(rb::StrongholdReplayBuffer, i=1)
    (i > MAX_STEPS || rb.actions[i] == -1) && return nothing
    return (rb.states[i], rb.actions[i], rb.rewards[i], rb.states[i+1]), i + 1
end

function update_buffer!(buffer::CircularVectorBuffer{StrongholdReplayBuffer}, env, p, M)
    env.resettable = true
    for _ in 1:M
        ep = Episode(env, p)
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

# This is a significant speedup over single-threaded
function update_buffer!(buffer::CircularVectorBuffer{StrongholdReplayBuffer}, _, p::MultiThreadPolicy, M)
    temp = SizedVector{M, StrongholdReplayBuffer}(undef)
    Threads.@threads for i in 1:M
        env = StrongholdEnvironment()
        if typeof(policy(p)) <: AbstractOraclePolicy
            policy(p).env = env
        end
        ep = Episode(env, p)
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

@time begin
    buffer_size = 10000
    replay_buffer = CircularVectorBuffer{StrongholdReplayBuffer}(buffer_size)
    rp = RandomPolicy()
    best_dev_winrate = winrate(MultiThreadPolicy(qa), dev)
    update_buffer!(replay_buffer, env, MultiThreadPolicy(rp), buffer_size)
end

begin
    eg = EpsilonGreedyPolicy(0.02, qa)
    #oo = OccasionalOraclePolicy(0.006, eg)
    #oo = OccasionalOraclePolicy(1, FirstExitPolicy())
    Reinforce.maxsteps(env::StrongholdEnvironment) = MAX_STEPS
    update_buffer!(replay_buffer, env, MultiThreadPolicy(eg), buffer_size)
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

# Sadly, this is not a significant speedup over single-threaded for batch sizes > 4.
# Zygote gradients use a ton of memory, either because of the way I'm calculating them or just because that's the way it is.
# Because of this, most of the time during training is spent on gc, which cannot be multithreaded (or already is, I forget).
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
    
    # combine gradients
    grads = temp[1]
    vs = collect.(values.(temp))
    for v in vs[2:end]
        vs[1] .+= v
    end

    Flux.Optimise.update!(opt, params, grads)
end

# Q-learning algorithm from https://www.cs.upc.edu/~mmartin/URL/Lecture4.pdf, slide 10
function training_loop(M, K, B, update_interval, qa::QApproximatorPolicy, eg, model, opt, replay_buffer::CircularVectorBuffer{StrongholdReplayBuffer})
    global best_dev_winrate, dev, iters
    println("Start\t", best_dev_winrate, "\t", now())
    next_time = now() + update_interval
    #oracle_batch = CircularVectorBuffer{StrongholdReplayBuffer}(B)
    #oracle = MultiThreadPolicy(OraclePolicy(nothing))
    params = Flux.params(model)
    while true
        iters += 1
        qa.target = deepcopy(qa.approximator)
        update_buffer!(replay_buffer, 0, MultiThreadPolicy(eg), M)
        
        for i = 1:K
            batch = rand(replay_buffer, B)
            train_step!(params, opt, batch, qa)
        end
        #=
        # This was an experiment. Having it use some of the oracle seems to tend to make it worse.
        # It can be used for fine-tuning with some strange results, notably that it will tend to "dive deep" once it reaches depth 12,
        # since the model detects that the oracle is probably running because it would never go that deep on its own.
        update_buffer!(oracle_batch, 0, oracle, B)
        train_step!(params, opt, oracle_batch, qa)
        =#
        mt = MultiThreadPolicy(qa)
        # This is used to speed up training, since calculating winrate for the whole dev set takes quite a while.
        # This way, we only calculate the whole dev set winrate when it's "promising".
        dev_subset = rand(dev, 100)
        subset_winrate = winrate(mt, dev_subset)
        
        if now() > next_time
            println(iters, "\t", subset_winrate, "\t", now())
            next_time = now() + update_interval
        end

        if subset_winrate > 0 && subset_winrate >= best_dev_winrate
            w = winrate(mt, dev)
            println(iters, "\t", subset_winrate, "\t", w)
            if w > best_dev_winrate
                println("New best! +", w - best_dev_winrate)
                best_dev_winrate = w
                bson("models/tmp/rl_rnn_a2_$iters.bson", rl_model = model, adam = opt, iters = iters)
            end
        end
    end
end

begin
    update_interval = Minute(10)
    M = 100  # number of replays to be added to the buffer during each step
    K = 100  # number of batches to train on during each step
    B = M ÷ K  # batch size
    training_loop(M, K, B, update_interval, qa, eg, model, opt, replay_buffer)
end

begin  # When you interrupt it and it would have maybe saved, do this
    w = winrate(qa, dev)
    println(iters, "\t", w)
    if w > best_dev_winrate
        println("New best! +", w - best_dev_winrate)
        best_dev_winrate = w
        bson("models/tmp/rl_recurrent_e_$iters.bson", rl_model = model, adam = opt, iters = iters)
    end
end

@time begin
    println(winrate(MultiThreadPolicy(qa), dev))
    println(winrate(MultiThreadPolicy(nb), dev))
    println(winrate(MultiThreadPolicy(eg), dev))
end

# How many steps does it take to win, if it does?
function win_distribution(policy, dataset, keep_losses=false)
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
                println(i, "\t", env.steps)
                max_steps_win = env.steps
            end
        elseif keep_losses
            push!(distribution, -maxsteps(env))
        end
    end
    return distribution
end

begin
    Reinforce.maxsteps(env::StrongholdEnvironment) = 100
    distr = win_distribution(bqa, dev)
    y_graph = [sum(distr .== x) for x in 1:maxsteps(env)]
    y_graph_cumulative = [sum(y_graph[1:x]) for x in 1:maxsteps(env)] ./ length(dev)
end

begin
    r = 1:100#maxsteps(env)
    Plots.bar(r, y_graph[r])
    xlabel!("# Steps to find portal room")
    ylabel!("Count")
    Plots.savefig("win_distribution.png")
end

begin
    Plots.bar(r, y_graph_cumulative[r])
    xlabel!("# Steps to find portal room")
    ylabel!("Success rate (cumulative)")
    Plots.savefig("win_cumulative.png")
end

begin
    best_model = BSON.load("models/rl_rnn_2.3.bson")[:rl_model]
    bqa = QApproximatorPolicy(best_model, best_model, 0.95)
    bnb = NoBacktrackingPolicy(bqa)
    beg = EpsilonGreedyPolicy(0.02, bqa)
end

begin
    Reinforce.maxsteps(env::StrongholdEnvironment) = 50
    println(winrate(MultiThreadPolicy(bqa), dev))
    #println(winrate(MultiThreadPolicy(bnb), dev))
    println(winrate(MultiThreadPolicy(beg), dev))
end

# This is my theoretical difficulty calculation for q-learning models.
# It's fairly simple, you just just add up how much the model doesn't want to go the correct direction.
function compute_difficulty(env, qa, verbose=false)
    oracle = OraclePolicy(env)
    ep = Episode(env, oracle)
    reset!(qa)
    total_difficulty = 0
    for (s, a, r, s′) in ep
        res = qa.approximator(s)
        correct = res[a + 1]
        total_difficulty += sum(filter(>(0), res .- correct))
        if verbose
            println(a, "\t", sum(filter(>(0), res .- correct)), "\t", res)
        end
    end
    return total_difficulty
end

begin
    difficulties = @showprogress 1 [compute_difficulty(StrongholdEnvironment(s, false), bqa) for s in strongholds[train]]
    wins = win_distribution(bqa, train, true)
end

begin
    f(x) = filter(<(100), x)
    win_difficulties = difficulties[wins .>= 0]
    loss_difficulties = difficulties[wins .< 0]
    #histogram2d(rand(f(loss_difficulties), length(difficulties)), rand(f(win_difficulties), length(difficulties)), normalize=:probability, aspect_ratio=:equal)
    histogram(difficulties ./ wins, normalize=:probability)
end


begin
    Flux.reset!(best_model)
    env = StrongholdEnvironment(strongholds[train[17491]], false)
    s = state(env)
    output = best_model(s)
    println(argmax(output) - 1, "\t", output)
end

begin
    r, s = step!(env, s, argmax(output) - 1)
    @show r, s
    output = best_model(s)
    println(argmax(output) - 1, "\t", output)
end