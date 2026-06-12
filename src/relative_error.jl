"""
    relative_error(predicted, ground_truth)

Calculate the relative error ||̂u - u||₂ / ||u||₂.
"""
function relative_error(predicted, ground_truth)
    L2 = u -> norm(u, 2)
    num = mapslices(L2, predicted .- ground_truth, dims = 1)
    den = mapslices(L2, ground_truth, dims = 1)
    return num ./ den
end

function get_mean_relative_state_error(
    predicted::AbstractArray{T,3},
    ground_truth::AbstractArray{T,3},
) where {T}
    rel_err = relative_error(predicted, ground_truth)
    return vec(mean(rel_err, dims = 3))
end

function get_mean_relative_constraint_error(
    system::AbstractDynamicalSystem{T},
    predicted::AbstractArray{T,3},
    ground_truth::AbstractArray{T,3},
) where {T}
    L2 = u -> norm(u, 2)
    g1 = ProjectedNDEs.constraints(predicted, nothing, system)
    g2 = ProjectedNDEs.constraints(ground_truth, nothing, system)

    return vec(mean(mapslices(L2, g1 .- g2, dims = 1), dims = 3))
end
