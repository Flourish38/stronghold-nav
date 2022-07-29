st = (x -> (x.correctDirection, x.targetDistance)).(vcat(strongholds...))
fs = first.(st)
ss = (x -> x[2]).(st)
plots = []
for dir in 0:5
    sub = ss[fs .== dir .&& ss .> 1]
    m = maximum(sub) + (dir == 0)
    push!(plots, (sum(sub) / length(sub), histogram(sub, nbins=m)))
end