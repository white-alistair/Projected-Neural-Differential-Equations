function select_last_dim(A, i)
    return selectdim(A, ndims(A), i)
end

function multiple_shooting(
    time_series::TimeSeries{T},
    constraints::AbstractArray{T};
    chunk_size::Int,
) where {T}
    (; times, trajectory) = time_series
    start_indexes = collect(1:chunk_size:length(time_series)-chunk_size)
    return @views [
        (
            times[i:i+chunk_size],
            select_last_dim(trajectory, i:i+chunk_size),
            select_last_dim(constraints, i:i+chunk_size),
        ) for i in start_indexes
    ]
end
