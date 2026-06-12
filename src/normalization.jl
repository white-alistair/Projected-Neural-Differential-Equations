struct Normalization{T,M<:AbstractMatrix{T}}
    μ::M
    σ::M
    ε::T
end

# Outer constructor with default epsilon
Normalization(μ::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T} = Normalization(μ, σ, T(1.0f-7))

function normalize(trajectories, norm::Normalization)
    (; μ, σ, ε) = norm
    return (trajectories .- μ) ./ (σ .+ ε)
end

function normalize(ts::TimeSeries, norm::Normalization)
    return TimeSeries(ts.times, normalize(ts.trajectory, norm))
end

function denormalize(normalized_trajectories, norm::Normalization)
    (; μ, σ, ε) = norm
    return normalized_trajectories .* (σ .+ ε) .+ μ
end

function denormalize(ts::TimeSeries, norm::Normalization)
    return TimeSeries(ts.times, denormalize(ts.trajectory, norm))
end

# No-op methods for when normalization is nothing
function normalize(data, ::Nothing)
    return data
end

function denormalize(data, ::Nothing)
    return data
end

# For serialization
Adapt.adapt_structure(to, x::Normalization) = Normalization(adapt(to, x.μ), adapt(to, x.σ), x.ε)
