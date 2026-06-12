function get_initial_conditions(target_trajectory, constraints)
    u0 = copy(selectdim(target_trajectory, ndims(target_trajectory) - 1, 1))
    g0 = copy(selectdim(constraints, ndims(constraints) - 1, 1))
    return u0, g0
end

function train!(
    model::AbstractModel{L,P,S,N},
    (; train_data, valid_data, test_data)::NamedTuple,
    epochs::Int,
    opt_state::Optimisers.Leaf,
    scheduler::ParameterSchedulers.AbstractSchedule,
    rng::Random.AbstractRNG,
    logger::TensorBoardLogger.TBLogger;
    # Loss function
    loss::F1 = MSE,
    # Solver args
    solver::SciMLBase.AbstractDEAlgorithm = Tsit5(),
    adjoint::SciMLSensitivity.AbstractAdjointSensitivityAlgorithm = BacksolveAdjoint(;
        autojacvec = ZygoteVJP(),
    ),
    reltol = 1.0e-4,
    abstol = 1.0e-4,
    log = true,
    verbose = false,
    manual_gc = false,
    # Two-stage training
    switch_epoch::Union{Int,Nothing} = nothing,
    switch_model_fn::Union{Function,Nothing} = nothing,
    switch_lr::Union{AbstractFloat,Nothing} = nothing,
    optimiser_rule = nothing,
    switch_optimiser::Bool = false,
) where {T,L,P<:ComponentVector{T},S,N,F1}
    @info "Beginning training..."

    # Initial setup
    θ = model.params
    θ_min = deepcopy(θ)
    min_val_loss = typemax(T)
    min_val_epoch = 0

    training_start_time = time()
    global_iter = 0
    model_switched = false
    for (epoch, learning_rate) in zip(1:epochs, scheduler)
        # Check for model switch (two-stage training)
        if !model_switched && switch_epoch !== nothing && epoch == switch_epoch && switch_model_fn !== nothing
            @info "[$(now(UTC))] Switching model at epoch $epoch"
            model = switch_model_fn(model.params)
            θ = model.params
            model_switched = true
            if switch_optimiser && optimiser_rule !== nothing
                @info "[$(now(UTC))] Re-initializing optimiser"
                opt_state = Optimisers.setup(optimiser_rule, θ)
            end
            if switch_lr !== nothing
                @info "[$(now(UTC))] Switching learning rate to $switch_lr"
            end
        end

        # Apply learning rate (handle switch_lr override after switch)
        lr_to_use = (model_switched && switch_lr !== nothing) ? T(switch_lr) : learning_rate
        Optimisers.adjust!(opt_state, lr_to_use)

        iter = 0
        training_losses = T[]
        grad_norms = T[]
        epoch_start_time = time()

        for (times, target_trajectory, constraints) in shuffle(rng, train_data)
            iter += 1
            global_iter += 1
            u0, g0 = get_initial_conditions(target_trajectory, constraints)

            training_loss, gradients = Zygote.withgradient(θ) do θ
                predicted_trajectory =
                    model(u0, g0, times, θ, solver; sensealg = adjoint, reltol, abstol)
                return loss(predicted_trajectory, target_trajectory)
            end

            push!(training_losses, training_loss)

            # Log gradient norm before clipping/update
            grad_norm = LinearAlgebra.norm(gradients[1])
            push!(grad_norms, grad_norm)
            TensorBoardLogger.log_value(logger, "iter/grad_norm", grad_norm, step = global_iter)
            TensorBoardLogger.log_value(logger, "iter/train_loss", training_loss, step = global_iter)

            opt_state, θ = Optimisers.update!(opt_state, θ, gradients[1])

            if log && verbose
                @info "[$(now(UTC))] " * @sprintf "[epoch = %04i] [iter = %04i] Loss = %.2e\n" epoch iter training_loss
            end
        end

        val_loss = evaluate(model, valid_data, loss, solver, reltol, abstol)
        epoch_duration = time() - epoch_start_time

        TensorBoardLogger.log_value(logger, "epoch/duration", epoch_duration, step = epoch)
        TensorBoardLogger.log_value(
            logger,
            "epoch/learning_rate",
            lr_to_use,
            step = epoch,
        )
        TensorBoardLogger.log_value(
            logger,
            "epoch/train_loss",
            mean(training_losses),
            step = epoch,
        )
        TensorBoardLogger.log_value(logger, "epoch/val_loss", val_loss, step = epoch)
        TensorBoardLogger.log_value(logger, "epoch/grad_norm_mean", mean(grad_norms), step = epoch)
        TensorBoardLogger.log_value(logger, "epoch/grad_norm_max", maximum(grad_norms), step = epoch)

        if log
            @info "[$(now(UTC))] " * @sprintf "[epoch = %04i] Learning rate = %.1e" epoch lr_to_use
            @info "[$(now(UTC))] " * @sprintf "[epoch = %04i] Train loss = %.2e\n" epoch mean(training_losses)
            @info "[$(now(UTC))] " * @sprintf "[epoch = %04i] Valid loss = %.2e\n" epoch val_loss
            @info "[$(now(UTC))] " * @sprintf "[epoch = %04i] Duration = %.1f seconds\n" epoch epoch_duration
            flush(stderr)
        end

        if val_loss < min_val_loss
            θ_min = deepcopy(θ)
            min_val_epoch = epoch
            min_val_loss = val_loss
        end

        # Manually call the GC to (hopefully) avoid OOM errors
        if manual_gc
            GC.gc(true)
            ccall(:malloc_trim, Cvoid, (Cint,), 0)
        end
    end
    training_duration = time() - training_start_time

    # Select the parameters that minimise the validation loss
    if !isempty(valid_data)
        θ .= θ_min
    end

    # Evaluate trained model
    test_loss = evaluate(model, test_data, loss, solver, reltol, abstol)
    TensorBoardLogger.log_value(logger, "loss/test_loss", test_loss, step = epochs)

    @info "[$(now(UTC))] Training complete."
    @info "[$(now(UTC))] " * @sprintf "Minimum validation loss = %.2e\n" min_val_loss
    @info "[$(now(UTC))] " * @sprintf "Test loss = %.2e\n" test_loss
    @info "[$(now(UTC))] " * @sprintf "Training duration = %.1f seconds\n" training_duration

    return training_duration, min_val_epoch, min_val_loss, test_loss, opt_state, rng
end
