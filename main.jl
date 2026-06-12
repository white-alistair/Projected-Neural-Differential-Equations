using InteractiveUtils
@info sprint(versioninfo)
@info "SLURM_NODELIST = $(get(ENV, "SLURM_NODELIST", nothing))"

using LibGit2
@info "HEAD = $(LibGit2.head("."))"

using ProjectedNDEs:
    parse_command_line,
    IEEE14Bus,
    NPendulum,
    QuarticPotential2D,
    LotkaVolterra,
    get_data,
    get_model,
    MSE,
    get_optimiser,
    get_scheduler,
    get_adjoint,
    train!,
    save_checkpoint,
    load_checkpoint,
    save_results,
    getstate,
    transform,
    MAGMACholesky
using Parameters,
    Random,
    FastClosures,
    OrdinaryDiffEq,
    SciMLBase,
    Lux,
    Optimisers,
    LinearAlgebra,
    TensorBoardLogger

function main(args)
    #! format: off
    # Unpack command line args into variables in current scope
    # Experiment args
    @unpack experiment, experiment_version, experiment_name = args
    @unpack device, job_id, checkpoint, rng_seed, NF = args
    # Data generation args
    @unpack dataset_name, t0, t1, dt, data_solver, data_reltol, data_abstol, normalize_data, filter_threshold = args
    # N-pendulum args
    @unpack N, friction = args
    # Data split args
    @unpack n_train, n_valid, n_test, chunk_size, batch_size = args
    # Solver args
    @unpack reltol, abstol, solver, sensealg, vjp, preconditioner = args
    # Neural net args
    @unpack hidden_layers, hidden_width, activation, use_skip = args
    # Stabilization args
    @unpack stabilization_param = args
    # Two-stage training args
    @unpack switch_epoch, switch_version, switch_lr, switch_optimiser = args
    # Optimizer args
    @unpack optimiser_rule, beta1, beta2, lambda, clip_grad = args
    # Training args
    @unpack loss, epochs, schedule_file, manual_gc, time_limit = args
    # I/0
    @unpack verbose, results_file = args
    #! format: on

    if device == :cuda
        device = MLDataDevices.CUDADevice()
    elseif device == :cpu
        device = MLDataDevices.CPUDevice()
    end

    rng = Random.Xoshiro(rng_seed)

    logger = TBLogger("experiments/$(experiment_name)/tensorboard/$(job_id)")

    # Log hyperparameters
    TensorBoardLogger.write_hparams!(
        logger,
        Dict{String,Any}(
            "layers" => hidden_layers,
            "width" => hidden_width,
            "activation" => string(activation),
            "batch_size" => batch_size,
            "optimiser_rule" => string(optimiser_rule),
            "lambda" => lambda,
            "beta1" => beta1,
            "beta2" => beta2,
            "clip_grad" => isnothing(clip_grad) ? "none" : clip_grad,
            "schedule" => schedule_file,
            "experiment_version" => string(experiment_version),
            "rng_seed" => rng_seed,
            "n_train" => n_train,
        ),
        ["loss/test_loss"],
    )

    if experiment == :ieee14bus
        system = IEEE14Bus{NF}(device)
    elseif experiment == :npendulum
        system = NPendulum{NF,N}(friction)
    elseif experiment == :quartic_potential_2d
        system = QuarticPotential2D{NF}()
    elseif experiment == :lotka_volterra
        system = LotkaVolterra{NF}()
    else
        error("Invalid experiment: $experiment")
    end
    experiment_version = Val(experiment_version)

    # Generate the training data
    data, normalization = get_data(
        system,
        experiment_version,
        n_train,
        n_valid,
        n_test;
        chunk_size,
        batch_size,
        dataset_name,
        t0,
        t1,
        dt,
        solver = data_solver,
        reltol = data_reltol,
        abstol = data_abstol,
        device,
        normalize_data,
        transform,
        rng,
        filter_threshold,
    )

    # Set up the solver backend
    backend = MAGMACholesky(; preconditioner)

    # Set up the model
    model = get_model(
        system,
        experiment_version,
        device,
        rng,
        activation,
        hidden_layers,
        hidden_width,
        normalization;
        γ = NF(stabilization_param),
        use_skip,
        backend,
    )

    # Count and log trainable parameters
    n_params = length(model.params)
    @info "Number of trainable parameters: $n_params"

    # Create switch model factory (for two-stage training)
    switch_model_fn = if switch_epoch !== nothing && switch_version !== nothing
        (θ) -> begin
            new_model = get_model(
                system,
                Val(switch_version),
                device,
                rng,
                activation,
                hidden_layers,
                hidden_width,
                normalization;
                γ = NF(stabilization_param),
                use_skip,
                backend,
            )
            new_model.params .= θ  # Transfer weights
            return new_model
        end
    else
        nothing
    end

    # Set up the loss function
    if loss == :MSE
        loss = MSE
    end

    # Set up the optimiser and the schedule
    optimiser_hyperparams = (; lambda, beta = (beta1, beta2))
    rule = get_optimiser(optimiser_rule, optimiser_hyperparams; clip_grad)
    opt_state = Optimisers.setup(rule, model.params)
    scheduler = get_scheduler(schedule_file)

    # Set up the adjoint
    adjoint = get_adjoint(sensealg, vjp)

    # Train the model
    training_duration, min_val_epoch, min_val_loss, test_loss, opt_state, rng = train!(
        model,
        data,
        epochs,
        opt_state,
        scheduler,
        rng,
        logger;
        loss,
        solver,
        adjoint,
        reltol,
        abstol,
        verbose,
        manual_gc,
        switch_epoch,
        switch_model_fn,
        switch_lr,
        optimiser_rule = rule,
        switch_optimiser,
    )

    save_checkpoint(job_id, model.params, opt_state, getstate(rng), normalization)

    # I/O
    save_results(
        results_file;
        job_id,
        system,
        experiment_version,
        hidden_layers,
        hidden_width,
        activation,
        use_skip,
        min_val_loss,
        test_loss,
        min_val_epoch,
        epochs,
        training_duration,
        time_limit,
        stabilization_param,
        t0,
        t1,
        dt,
        n_train,
        n_valid,
        n_test,
        filter_threshold,
        dataset_name,
        chunk_size,
        batch_size,
        optimiser_rule,
        optimiser_hyperparams = string(optimiser_hyperparams),
        clip_grad,
        schedule_file,
        switch_epoch,
        switch_version,
        switch_lr,
        switch_optimiser,
        reltol,
        abstol,
        rng_seed,
        NF,
        device,
        normalize_data,
        preconditioner,
        comment = "",
    )
end

args = parse_command_line(log = true)
main(args)
