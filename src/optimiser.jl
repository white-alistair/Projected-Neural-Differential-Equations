"""
    get_optimiser(rule_type, hyperparameters)
    
Helper function for setting up an optimiser object.

See Optimisers.jl for full details of the optimisers and their hyperparameters.
"""
function get_optimiser(rule_type, hyperparameters; clip_grad = nothing)
    if rule_type == :Adam
        rule = Optimisers.Adam()
    elseif rule_type == :AdamW
        rule = Optimisers.AdamW()
    else
        error("Unknown optimiser: $rule_type. Supported: :Adam, :AdamW")
    end

    if !isempty(hyperparameters)
        rule = Optimisers.adjust(rule; hyperparameters...)
    end

    if !isnothing(clip_grad)
        rule = Optimisers.OptimiserChain(Optimisers.ClipGrad(clip_grad), rule)
    end

    return rule
end
