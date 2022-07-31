# Here we go...

begin
    include("reinforcement_environment.jl")
    include("utils.jl")
    using CircularArrayBuffers
end

mutable struct QApproximatorPolicy <: AbstractModelPolicy
    model
    target
    γ
end
Reinforce.action(qa::QApproximatorPolicy, r, s, A) = A[argmax(qa.model(s))]
function Reinforce.reset!(qa::QApproximatorPolicy)
    Flux.reset!(qa.model)
    #Flux.reset!(qa.target)  # Not needed during deployment, so not used here
end
function loss(qa::QApproximatorPolicy, s, a, r, s′)
    target = r + qa.γ*maximum(qa.target(s′))
    prediction = qa.model(s)[a]
    (target - prediction)^2
end
# I feel like there *must* be a way to optimize this.
function loss(qa::QApproximatorPolicy, replay::StrongholdReplay)
    n = 0
    total = 0
    Flux.reset!(qa.model)
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
#=
# why is this slower. I hate this. AAA
function loss(qa::QApproximatorPolicy, replay::StrongholdReplay)
    Flux.reset!(qa.approximator)
    Flux.reset!(qa.target)
    n = length(replay)
    s, a, r = valid_data(replay, n)
    _ = qa.target(s[:, 1])
    mean((r[i] + qa.γ*maximum(qa.target(s[:, i+1])) - qa.approximator(s[:, i])[a[i]+1])^2 for i in 1:n)
end
=#
function loss(qa::QApproximatorPolicy, batch::A) where A<:AbstractArray{<:StrongholdReplay}
    total = 0
    for replay in batch
        total += loss(qa, replay)
    end
    return total / length(batch)
end