"""
    get_scheduler(schedule_path::String)

Helper function to set up a parameter scheduler from a schedule.toml file.
"""
function get_scheduler(schedule_path::String)
    schedule_dict = TOML.parsefile(schedule_path)
    return get_scheduler(schedule_dict)
end

function get_scheduler(schedule_dict::Dict)
    schedule_epoch_pairs =
        [unpack_schedule(config) for config in schedule_dict["schedules"]]
    return ParameterSchedulers.Sequence(schedule_epoch_pairs...)
end

"""
    unpack_schedule(config)

Given schedule config, create a schedule object.

See ParameterSchedulers.jl for more details.
"""
function unpack_schedule(config)
    @unpack epochs, schedule = config
    if schedule == "Constant"
        @unpack lr = config
        return ParameterSchedulers.Constant(lr) => epochs
    elseif schedule == "ExpDecay"
        @unpack max_lr, decay, epochs = config
        return ParameterSchedulers.Exp(; λ = max_lr, γ = decay) => epochs
    elseif schedule == "CosAnneal"
        @unpack min_lr, max_lr = config
        period = get(config, "period", config["epochs"])  # Use period if given, else epochs
        return ParameterSchedulers.CosAnneal(; λ0 = max_lr, λ1 = min_lr, period) => epochs
    elseif schedule == "LinearRamp"
        @unpack min_lr, max_lr, epochs = config
        return ParameterSchedulers.Triangle(;
            λ0 = max_lr,
            λ1 = min_lr,
            period = 2 * epochs,
        ) => epochs
    else
        error("Unknown schedule type: $(schedule). Supported: Constant, ExpDecay, CosAnneal, LinearRamp")
    end
end
