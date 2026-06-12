"""
    MSE(predicted_trajectory, target_trajectory)

Compute the mean-squared error between the predicted trajectory and the target trajectory.
Returns Inf if dimensions don't match (e.g., if ODE solver aborted early).
"""
function MSE(predicted_trajectory, target_trajectory)
    # Check for dimension mismatch (ODE solver may abort early)
    if size(predicted_trajectory) != size(target_trajectory)
        return eltype(predicted_trajectory)(Inf)
    end
    return mean(abs2, predicted_trajectory[:, :, 2:end] .- target_trajectory[:, :, 2:end])  # Do not include u0
end
