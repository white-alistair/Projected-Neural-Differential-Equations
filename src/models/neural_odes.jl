abstract type AbstractNeuralODE{L,P,S,N} <: AbstractModel{L,P,S,N} end

function (model::AbstractNeuralODE)(
    u0,
    g0,
    times,
    θ,
    solver = Tsit5();
    sensealg = adjoint,
    reltol = 1.0f-4,
    abstol = 1.0f-4,
)
    (; lux_model, state, normalization) = model

    rhs = @closure (u, θ, t) -> lux_model(u, θ, state)[1]

    u0 = denormalize(u0, normalization)  # no-op if normalization is nothing
    times = times[:, 1]
    tspan = (times[1], times[end])

    prob = ODEProblem{false,SciMLBase.FullSpecialize}(rhs, u0, tspan)
    sol = solve(prob, solver; p = θ, saveat = times, sensealg, reltol, abstol)
    result = stack(sol.u, dims = 2)

    return normalize(result, normalization)  # no-op if normalization is nothing
end

# FIRST ORDER AUTONOMOUS NEURAL ODE
struct FirstOrderAutonomousNeuralODE{L,P,S,N} <: AbstractNeuralODE{L,P,S,N}
    lux_model::L
    params::P
    state::S
    normalization::N
end

function FirstOrderAutonomousNeuralODE(
    dim,
    hidden_layers,
    hidden_width,
    activation,
    rng,
    T,
    device;
    preprocess_mlp_inputs = identity,
    postprocess_mlp_outputs = identity,
    normalization = nothing,
    use_skip = false,
    kwargs...,
)
    mlp = get_mlp(dim => dim, hidden_layers, hidden_width, activation; use_skip)
    layers = (Lux.WrappedFunction(preprocess_mlp_inputs), mlp)
    if postprocess_mlp_outputs !== identity
        layers = (layers..., Lux.WrappedFunction(postprocess_mlp_outputs))
    end
    lux_model = Lux.Chain(layers...)
    
    params, state = Lux.setup(rng, lux_model)
    if T == Float64
        params, state = (params, state) |> Lux.f64
    end
    params = ComponentArray(params) |> device
    state = state |> device

    return FirstOrderAutonomousNeuralODE(lux_model, params, state, normalization)
end

# SECOND ORDER AUTONOMOUS NEURAL ODE
struct SecondOrderAutonomousNeuralODE{L,P,S,N} <: AbstractNeuralODE{L,P,S,N}
    lux_model::L
    params::P
    state::S
    normalization::N
end

function SecondOrderAutonomousNeuralODE(
    dim,
    hidden_layers,
    hidden_width,
    activation,
    rng,
    T,
    device;
    mlp_input_dim = 2dim,
    preprocess_mlp_inputs = identity,
    normalization = nothing,
    use_skip = false,
)
    mlp = get_mlp(mlp_input_dim => dim, hidden_layers, hidden_width, activation; use_skip)
    lux_model = Lux.Parallel(
        vcat,
        Lux.WrappedFunction(u -> u[dim+1:end, :]),
        Lux.Chain(
            Lux.WrappedFunction(preprocess_mlp_inputs),
            mlp,
        ),
    )

    params, state = Lux.setup(rng, lux_model)
    if T == Float64
        params, state = (params, state) |> Lux.f64
    end
    params = ComponentArray(params) |> device
    state = state |> device

    return SecondOrderAutonomousNeuralODE(lux_model, params, state, normalization)
end
