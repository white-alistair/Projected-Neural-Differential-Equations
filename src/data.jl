function get_data(
    system,
    experiment_version,
    n_train,
    n_valid,
    n_test;
    chunk_size,
    batch_size,
    t0,
    t1,
    dt,
    solver,
    reltol,
    abstol,
    transform,
    device,
    normalize_data = true,
    dataset_name,
    rng,
    filter_threshold = nothing,  # Optional: filter chunks with κ(GGᵀ) > threshold
)
    # 1. Compute full trajectories
    train_trajectories = [
        get_trajectory(system, experiment_version; t0, t1, dt, solver, reltol, abstol, transform, device, rng)
        for _ = 1:n_train
    ]
    valid_trajectories =
        [get_trajectory(system, experiment_version; t0, t1, dt, solver, reltol, abstol, transform, device, rng) for _ = 1:n_valid]
    test_trajectories =
        [get_trajectory(system, experiment_version; t0, t1, dt, solver, reltol, abstol, transform, device, rng) for _ = 1:n_test]

    # 2. Compute constraints (before normalization)
    train_constraints = [get_constraints(system, experiment_version, ts; device) for ts in train_trajectories]
    valid_constraints = [get_constraints(system, experiment_version, ts; device) for ts in valid_trajectories]
    test_constraints = [get_constraints(system, experiment_version, ts; device) for ts in test_trajectories]

    # 3. Compute normalization stats and apply to trajectories
    if normalize_data
        normalization = compute_normalization_stats_from_trajectories(
            train_trajectories,
            valid_trajectories,
        )

        train_trajectories = [normalize(traj, normalization) for traj in train_trajectories]
        valid_trajectories = [normalize(traj, normalization) for traj in valid_trajectories]
        test_trajectories = [normalize(traj, normalization) for traj in test_trajectories]
    else
        normalization = nothing
    end

    # 4. Chunk and batch the trajectories and constraints
    # Only filter training data - validation and test should use full distribution
    filter_chunks_train = filter_threshold !== nothing ? (system, filter_threshold) : nothing
    
    train_data = process_trajectories_to_batches(
        train_trajectories,
        train_constraints,
        chunk_size,
        batch_size,
        rng;
        filter_chunks = filter_chunks_train,
    )
    valid_data = process_trajectories_to_batches(
        valid_trajectories,
        valid_constraints,
        chunk_size,
        batch_size,
        rng;
        filter_chunks = nothing,  # Don't filter validation
    )
    test_data = process_trajectories_to_batches(
        test_trajectories,
        test_constraints,
        chunk_size,
        batch_size,
        rng;
        filter_chunks = nothing,  # Don't filter test
    )

    return (; train_data, valid_data, test_data), normalization
end

# Specialized method for AbstractPowerGrid systems that loads pre-computed trajectories
function get_data(
    system::AbstractPowerGrid,
    experiment_version,
    n_train,
    n_valid,
    n_test;
    chunk_size,
    batch_size,
    dt,
    device,
    normalize_data,
    dataset_name,
    rng,
    kwargs...
)
    # Load pre-computed trajectories from JLD2 file
    if system isa IEEE14Bus
        file_path = "power_grids/ieee14bus/$(dataset_name).jld2"
    else
        error("Unsupported AbstractPowerGrid subtype: $(typeof(system))")
    end
    raw_trajectories = JLD2.load_object(file_path)  # Shape: (28, 101, 1000)
    
    # Create time array based on dt and data size
    n_time_points = size(raw_trajectories, 2)  # 101 time points
    times = collect((0:n_time_points-1) * dt)  # [0, dt, 2*dt, ..., 100*dt]
    
    # Total number of trajectories available
    total_trajectories = size(raw_trajectories, 3)
    
    # Check if we have enough trajectories
    total_needed = n_train + n_valid + n_test
    if total_needed > total_trajectories
        error("Not enough trajectories in dataset. Need $total_needed, but only have $total_trajectories")
    end
    
    # Split trajectories into train/valid/test sets
    train_trajectories = []
    valid_trajectories = []
    test_trajectories = []
    
    # Create TimeSeries objects for train set
    for i in 1:n_train
        traj = raw_trajectories[:, :, i]  # Shape: (28, 101)
        push!(train_trajectories, TimeSeries(times, traj, device, Float32))
    end
    
    # Create TimeSeries objects for validation set
    for i in (n_train + 1):(n_train + n_valid)
        traj = raw_trajectories[:, :, i]  # Shape: (28, 101)
        push!(valid_trajectories, TimeSeries(times, traj, device, Float32))
    end
    
    # Create TimeSeries objects for test set
    for i in (n_train + n_valid + 1):(n_train + n_valid + n_test)
        traj = raw_trajectories[:, :, i]  # Shape: (28, 101)
        push!(test_trajectories, TimeSeries(times, traj, device, Float32))
    end

    # 2. Compute constraints
    train_constraints = [get_constraints(system, experiment_version, ts; device) for ts in train_trajectories]
    valid_constraints = [get_constraints(system, experiment_version, ts; device) for ts in valid_trajectories]
    test_constraints = [get_constraints(system, experiment_version, ts; device) for ts in test_trajectories]

    # 3. Compute normalization stats and apply to trajectories
    if normalize_data
        normalization = compute_normalization_stats_from_trajectories(
            train_trajectories,
            valid_trajectories,
        )

        # Set any standard deviations less than 1e-6 to 1 to prevent numerical issues
        normalization.σ .= map(x -> x < 1f-4 ? 1f0 : x, normalization.σ)

        train_trajectories = [normalize(traj, normalization) for traj in train_trajectories]
        valid_trajectories = [normalize(traj, normalization) for traj in valid_trajectories]
        test_trajectories = [normalize(traj, normalization) for traj in test_trajectories]
    else
        normalization = nothing
    end

    # 4. Chunk and batch the trajectories with pre-computed physical-space constraints
    train_data = process_trajectories_to_batches(
        train_trajectories,
        train_constraints,
        chunk_size,
        batch_size,
        rng,
    )
    valid_data = process_trajectories_to_batches(
        valid_trajectories,
        valid_constraints,
        chunk_size,
        batch_size,
        rng,
    )
    test_data = process_trajectories_to_batches(
        test_trajectories,
        test_constraints,
        chunk_size,
        batch_size,
        rng,
    )

    return (; train_data, valid_data, test_data), normalization
end

function get_trajectory(
    system::AbstractDynamicalSystem{T},
    experiment_version;
    t0,
    t1,
    dt,
    solver,
    reltol,
    abstol,
    transform,
    device,
    rng,
) where {T}
    u0 = initial_conditions(system, rng)  # Random initial conditions using provided RNG
    tspan = (t0, t1)
    saveat = t0:dt:t1
    prob = ODEProblem(system, u0, tspan)
    sol = solve(prob, solver; saveat, reltol, abstol)
    traj = stack(sol.u, dims = 2)
    traj = transform(Array(traj), system, experiment_version)
    return TimeSeries(sol.t, traj, device, T)
end

function compute_normalization_stats_from_trajectories(
    train_trajectories,
    valid_trajectories,
)
    # Stack all trajectory data from TimeSeries objects
    all_trajectories = []
    for ts in vcat(train_trajectories, valid_trajectories)
        push!(all_trajectories, ts.trajectory)
    end

    # Compute mean and std across the time and trajectory dimensions
    stacked_trajectories = stack(all_trajectories)
    μ = mean(stacked_trajectories; dims = (2,3))
    μ = dropdims(μ; dims = 3)
    σ = std(stacked_trajectories; dims = (2,3))
    σ = dropdims(σ; dims = 3)

    return Normalization(μ, σ)
end

function process_trajectories_to_batches(
    trajectories,
    constraints_list,
    chunk_size,
    batch_size,
    rng;
    filter_chunks = nothing,  # Optional: (system, κ_threshold) tuple for filtering
)
    all_chunks = []
    for (ts, ts_constraints) in zip(trajectories, constraints_list)
        # Create chunks using multiple_shooting
        chunks = multiple_shooting(ts, ts_constraints; chunk_size)
        append!(all_chunks, chunks)
    end
    
    # Optionally filter chunks by condition number
    if filter_chunks !== nothing
        system, κ_threshold = filter_chunks
        n_before = length(all_chunks)
        all_chunks = filter(c -> is_chunk_well_conditioned(c, system, κ_threshold), all_chunks)
        n_after = length(all_chunks)
        @info "Chunk filtering: kept $n_after / $n_before chunks (removed $(n_before - n_after), threshold κ < $κ_threshold)"
    end
    
    shuffle!(rng, all_chunks)
    return batch_chunks(all_chunks, batch_size)
end

"""
    is_chunk_well_conditioned(chunk, system, κ_threshold)

Check if all time points in a chunk have condition number κ(GGᵀ) < κ_threshold.
Returns true if the chunk is well-conditioned (should be kept).
"""
function is_chunk_well_conditioned(chunk, system, κ_threshold)
    times, trajectory, constraints = chunk
    n_times = size(trajectory, 2)
    
    for t_idx in 1:n_times
        u = trajectory[:, t_idx:t_idx]  # (state_dim, 1)
        G = constraints_jacobian(u, times[t_idx], system)
        G_mat = G[:, :, 1]  # (m, n)
        GGt = G_mat * G_mat'  # (m, m)
        
        # Compute condition number
        eigvals_GGt = eigvals(Symmetric(Array(GGt)))
        λ_min = minimum(eigvals_GGt)
        λ_max = maximum(eigvals_GGt)
        κ = λ_max / max(λ_min, eps(eltype(GGt)))
        
        if κ > κ_threshold
            return false
        end
    end
    return true
end

function batch_chunks(chunks, batch_size)
    batched_data = MLUtils.chunk(chunks; size = batch_size)
    return map(batched_data) do batch
        times = stack([item[1] for item in batch])
        trajectories = stack([item[2] for item in batch])
        constraints = stack([item[3] for item in batch])
        (times, trajectories, constraints)
    end
end

function transform(traj, system, experiment_version)
    return traj  # Default transformation is no-op
end

function get_constraints(system, experiment_version, ts::TimeSeries{T}; device) where {T}
    return get_constraints(system, experiment_version, ts.trajectory, ts.times; device)
end

function get_constraints(
    system,
    experiment_version,
    trajectory::AbstractArray{T},
    times::AbstractVector{T};
    device,
) where {T}
    return device(constraints(trajectory, times, system))
end
