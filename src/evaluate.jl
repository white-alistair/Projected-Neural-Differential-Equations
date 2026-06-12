"""
    evaluate(θ, data, loss, solver, reltol, abstol)

Evaluate the loss for the parameters θ on the given data, i.e. validation or test data.
"""
function evaluate(model, data, loss, solver, reltol, abstol)
    if isnothing(data)
        return NaN64
    end

    losses = Float64[]
    for (times, target_trajectory, constraints) in data
        u0, g0 = get_initial_conditions(target_trajectory, constraints)
        predicted_trajectory =
            model(u0, g0, times, model.params, solver; sensealg = nothing, reltol, abstol)
        push!(losses, loss(predicted_trajectory, target_trajectory))
    end

    return mean(losses)
end
