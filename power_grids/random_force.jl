using Distributions

function random_force_uniform_circle(a, b, N)
    v = sqrt.(rand(Uniform(a, b), N))
    θ = 2π .* rand(N)
    return collect(Iterators.flatten(zip(v .* cos.(θ), v .* sin.(θ))))
end
