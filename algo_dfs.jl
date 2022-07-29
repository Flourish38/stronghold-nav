@time begin
    #const INPUT_VEC_ORDER = "DEPTH,DIRECTION,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5"  # stateless non-reinforcement
    const INPUT_VEC_ORDER = "DEPTH,DIRECTION,ENTRY,PREV_EXIT_INDEX_COMPAT,CURRENT,PARENT_ROOM,EXIT_1,EXIT_2,EXIT_3,EXIT_4,EXIT_5"
    include("reinforcement_environment.jl")
    include("utils.jl")
end

@time begin
    using Reinforce
    using StaticArrays
    using ProgressMeter
    using UnicodePlots
    using Flux
    using BSON
end

begin
    struct DFSPolicy{N} <: AbstractPolicy
        checked::Set{Any}
        current::MVector{N, Int8}
        function DFSPolicy(n)
            new{n}(Set{Any}(), @MVector zeros(Int8, n))
        end
    end
    Reinforce.reset!(π::DFSPolicy) = empty!(π.checked)
    function Reinforce.action(π::DFSPolicy{N}, r, s, A) where N
        d = Int(s[SCALAR_FUNCTION_OFFSETS[findfirst(==("DEPTH"), INPUT_VEC_FORMAT)]])
        if d == N
            push!(π.checked, π.current[SOneTo(d)])
            return 0
        end
        j = VECTOR_FUNCTION_OFFSETS[findfirst(==("PARENT_ROOM"), INPUT_VEC_FORMAT) - NUM_SCALAR_FUNCTIONS] + 10  # idx of first stupid room
        for i in 1:5
            π.current[d+1] = i
            if all(s[j+14i:j+4+14i] .!= 1) && !(π.current[SOneTo(d+1)] in π.checked)
                return i
            end
        end
        push!(π.checked, π.current[SOneTo(d)])
        return 0
    end
end

p = NeverMissTargetPolicy(DFSPolicy(8))
mt = MultiThreadPolicy(p)

Reinforce.maxsteps(env::StrongholdEnvironment) = 150
wr_dfs = vcat(zeros(4), @showprogress [winrate(MultiThreadPolicy(NeverMissTargetPolicy(DFSPolicy(i))), dev) for i in 5:34])

begin
    mutable struct ComplexityDFS{N} <: AbstractPolicy
        checked::Set{Any}
        current::MVector{N, Int8}
        complexities::Vector{Int8}
        complexity_cap::Int
        complexity_min_depth::Int
        function ComplexityDFS(n, m, md)
            new{n}(Set{Any}(), @MVector(zeros(Int8, n)), zeros(Int, n+1), m, md)
        end
    end
    Reinforce.reset!(π::ComplexityDFS) = π.checked = empty!(π.checked)
    function Reinforce.action(π::ComplexityDFS{N}, r, s, A) where N
        d = Int(s[SCALAR_FUNCTION_OFFSETS[findfirst(==("DEPTH"), INPUT_VEC_FORMAT)]])
        if d == N
            push!(π.checked, π.current[SOneTo(d)])
            return 0
        end
        j = VECTOR_FUNCTION_OFFSETS[findfirst(==("PARENT_ROOM"), INPUT_VEC_FORMAT) - NUM_SCALAR_FUNCTIONS] + 10  # idx of first stupid room
        valid = 0
        first_valid = 0
        for i in 1:5
            if all(s[j+14i:j+4+14i] .!= 1)  # not a stupid room
                valid += 1
                if first_valid == 0
                    π.current[d+1] = i
                    if !(π.current[SOneTo(d+1)] in π.checked) && first_valid == 0
                        first_valid = i
                    end
                end
            end
        end
        π.complexities[d+1] = valid
        if π.complexity_cap < prod(π.complexities[π.complexity_min_depth:d+1]) || first_valid == 0
            push!(π.checked, π.current[SOneTo(d)])
            return 0
        end
        return first_valid
    end
end

p = NeverMissTargetPolicy(ComplexityDFS(7, 20, 4))
mt = MultiThreadPolicy(p)

ws_dfsc = vcat(ws_dfsc, @showprogress [[vcat(zeros(4), [winrate(MultiThreadPolicy(NeverMissTargetPolicy(ComplexityDFS(i, m, md))), dev) for i in 5:15]) for m in 2:30] for md in 7:8])
ws_dfsc_r = ws_dfsc[argmax([maximum(maximum.(x)) for x in ws_dfsc])]

for (s, a, r, s′) in Episode(StrongholdEnvironment(), p)
    println(a, "\t", s[1], "\t", r)
end

begin
    struct InformedDFS <: AbstractPolicy
        max::Int
        min::Int
        checked::Dict{Any, Float64}
        current
        model
        function InformedDFS(model, max, min=0)
            new(max, min, Dict{Any, Float64}(), @MVector(zeros(Int8, max)), model)
        end
    end
    function Reinforce.reset!(π::InformedDFS) 
        empty!(π.checked)
        Flux.reset!(π.model)
    end
    function Reinforce.action(π::InformedDFS, r, s, A)
        d = Int(s[SCALAR_FUNCTION_OFFSETS[findfirst(==("DEPTH"), INPUT_VEC_FORMAT)]])
        if d == π.max
            π.checked[π.current[SOneTo(d)]] = Inf
            return 0
        end
        m = π.model(s)
        # for dist model
        #=
        m_ = Any[0, 0, 0, 0, 0]
        for i = 1:5
            π.current[d+1] = i
            j = 0
            if haskey(π.checked, π.current[SOneTo(d+1)]) && false
                j = Int(clamp(π.checked[π.current[SOneTo(d+1)]], 0, 8))
            end
            m_[i] = sum(m[(25+j+8(i-1)):(32+8(i-1))] ./ ((1+j):8))
        end
        
        m = vcat(sum(m[1:24] ./ (1:24)), m_)
        =#

        # only use this if you're using a model that can backtrack
        #=
        if argmax(m) == 1 && d >= π.min
            
            full_checked = true
            for i in 1:5
                π.current[d+1] = i
                if !haskey(π.checked, π.current[SOneTo(d+1)]) || π.checked[π.current[SOneTo(d+1)]] != Inf
                    full_checked = true
                    break
                end
            end
            if full_checked
                π.checked[π.current[SOneTo(d)]] = Inf
            else
                max_checked = 0
                for i in 1:5
                    π.current[d+1] = i
                    if haskey(π.checked, π.current[SOneTo(d+1)]) && max_checked < π.checked[π.current[SOneTo(d+1)]] && π.checked[π.current[SOneTo(d+1)]] != Inf
                        max_checked = π.checked[π.current[SOneTo(d+1)]]
                    end
                end
                π.checked[π.current[SOneTo(d)]] = max_checked + 1
            end
            
            return 0
        end
        =#
        
        j = VECTOR_FUNCTION_OFFSETS[findfirst(==("PARENT_ROOM"), INPUT_VEC_FORMAT) - NUM_SCALAR_FUNCTIONS] + 10  # idx of first stupid room
        m = m[2:6]
        for i in sort(1:5, by=x -> m[x], rev=true)
            π.current[d+1] = i
            if all(s[j+14i:j+4+14i] .!= 1) && (!haskey(π.checked, π.current[SOneTo(d+1)]) || π.checked[π.current[SOneTo(d+1)]] != Inf)
                return i
            end
        end
        π.checked[π.current[SOneTo(d)]] = Inf
        return 0
    end
end

#model = BSON.load("models/stateless_185.bson")[:stateless_model]
model = BSON.load("models/rl_rnn_2.2.bson")[:rl_model]
model = BSON.load("models/stateless_dist_188.bson")[:stateless_dist]

p = NeverMissTargetPolicy(InformedDFS(model, 10))
mt = MultiThreadPolicy(ans)

Reinforce.maxsteps(env::StrongholdEnvironment) = 755
wr_idfs = [vcat(zeros(4), @showprogress [winrate(MultiThreadPolicy(NeverMissTargetPolicy(InformedDFS(model, max, min))), dev) for max in 5:15]) for min in 1:10]

begin
    Reinforce.maxsteps(::StrongholdEnvironment) = 589
    println(winrate(NeverMissTargetPolicy(DFSPolicy(33)), dev), "\t", winrate(NeverMissTargetPolicy(InformedDFS(33, model)), dev))
end
