struct TimeSeries{T,V<:AbstractVector{T},A<:AbstractArray{T}}
    times::V
    trajectory::A
    TimeSeries(times::AbstractVector{T}, trajectory::AbstractArray{T}) where {T} =
        size(times)[end] != size(trajectory)[end] ?
        throw(DimensionMismatch("number of times and observations do not match")) :
        new{T,typeof(times),typeof(trajectory)}(times, trajectory)
end

function TimeSeries(
    times::AbstractVector{T},
    trajectory::AbstractArray{T},
    device::MLDataDevices.AbstractDevice,
    NF = T,  # Optional conversion
) where {T}
    TimeSeries(NF.(times), device(NF.(trajectory)))
end

Base.length(time_series::TimeSeries) = length(time_series.times)
