abstract type AbstractStabilizedModel{L,P,S,N} <: AbstractModel{L,P,S,N} end

function (model::AbstractStabilizedModel)(
    u0,
    g0,
    times,
    θ,
    solver = Tsit5();
    sensealg = adjoint,
    reltol = 1.0f-4,
    abstol = 1.0f-4,
)
    (; lux_model, state, normalization, γ, system, backend) = model

    rhs = @closure (u, θ, t) -> begin
        du_dt = lux_model(u, θ, state)[1]
        G = constraints_jacobian(u, t, system)
        g = constraints(u, t, system) .- g0
        return du_dt .- γ * min_norm_solve(G, g, backend)
    end

    u0 = denormalize(u0, normalization)  # no-op if normalization is nothing
    times = times[:, 1]
    tspan = (times[1], times[end])

    prob = ODEProblem{false,SciMLBase.FullSpecialize}(rhs, u0, tspan)
    sol = solve(prob, solver; p = θ, saveat = times, sensealg, reltol, abstol)
    result = stack(sol.u, dims = 2)

    return normalize(result, normalization)  # no-op if normalization is nothing
end

# FIRST ORDER AUTONOMOUS STABILIZED NEURAL ODE
struct FirstOrderAutonomousStabilizedNeuralODE{L,P,S,N,Sys,B,G} <:
       AbstractStabilizedModel{L,P,S,N}
    lux_model::L
    params::P
    state::S
    normalization::N
    system::Sys
    backend::B
    γ::G
end

function FirstOrderAutonomousStabilizedNeuralODE(
    dim,
    hidden_layers,
    hidden_width,
    activation,
    rng,
    T,
    device,
    system::AbstractDynamicalSystem,
    γ;
    preprocess_mlp_inputs = identity,
    postprocess_mlp_outputs = identity,
    normalization = nothing,
    backend = MAGMACholesky(),
    use_skip = false,
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

    return FirstOrderAutonomousStabilizedNeuralODE(
        lux_model,
        params,
        state,
        normalization,
        system,
        backend,
        γ,
    )
end

# SECOND ORDER AUTONOMOUS STABILIZED NEURAL ODE
struct SecondOrderAutonomousStabilizedNeuralODE{L,P,S,N,Sys,B,G} <:
       AbstractStabilizedModel{L,P,S,N}
    lux_model::L
    params::P
    state::S
    normalization::N
    system::Sys
    backend::B
    γ::G
end

function SecondOrderAutonomousStabilizedNeuralODE(
    dim,
    hidden_layers,
    hidden_width,
    activation,
    rng,
    T,
    device,
    system::AbstractDynamicalSystem,
    γ;
    mlp_input_dim = 2dim,
    preprocess_mlp_inputs = identity,
    normalization = nothing,
    backend = MAGMACholesky(),
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

    return SecondOrderAutonomousStabilizedNeuralODE(
        lux_model,
        params,
        state,
        normalization,
        system,
        backend,
        γ,
    )
end
