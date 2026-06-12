function save_checkpoint(id, params, optimizer_state, rng_state, normalization; dir = "checkpoints")
    cpu_device = MLDataDevices.CPUDevice()
    mkpath(dir)
    file = "$(id).jld2"
    path = joinpath(dir, file)
    params = Adapt.adapt_structure(cpu_device, params)
    optimizer_state = Adapt.adapt_structure(cpu_device, destructure(optimizer_state)[1])
    normalization = Adapt.adapt_structure(cpu_device, normalization)
    jldsave(path; params, optimizer_state, rng_state, normalization)
    return nothing
end

function load_checkpoint(id; dir = "checkpoints", adapt_to = MLDataDevices.CPUDevice())
    mkpath(dir)
    file = "$(id).jld2"
    path = joinpath(dir, file)
    checkpoint = load(path)
    params = Adapt.adapt_structure(adapt_to, checkpoint["params"])
    optimizer_state = Adapt.adapt_structure(adapt_to, checkpoint["optimizer_state"])
    rng_state = checkpoint["rng_state"]
    normalization = Adapt.adapt_structure(adapt_to, checkpoint["normalization"])
    return params, optimizer_state, rng_state, normalization
end

# This function will be added to Random.jl in Julia 1.11
getstate(x::Random.Xoshiro) = (x.s0, x.s1, x.s2, x.s3, x.s4)
