"""
    get_adjoint(sensealg, vjp, checkpointing = false)

Helper function for setting up the adjoint sensitivity method.

See the SciMLSensitivity.jl docs for full details of the options.
"""
function get_adjoint(sensealg, vjp, checkpointing = false)
    if isnothing(sensealg)
        return nothing
    end

    if isnothing(vjp)
        autojacvec = nothing
    elseif vjp == :ZygoteVJP
        autojacvec = ZygoteVJP()
    elseif vjp == :ReverseDiffVJP
        autojacvec = ReverseDiffVJP(true)
    elseif vjp == :EnzymeVJP
        autojacvec = EnzymeVJP()  # Not currently supported by SciMLSensitivity for this problem type
    end
    
    if sensealg == :BacksolveAdjoint
        return BacksolveAdjoint(; autojacvec, checkpointing)
    elseif sensealg == :InterpolatingAdjoint
        return InterpolatingAdjoint(; autojacvec, checkpointing)
    elseif sensealg == :QuadratureAdjoint
        return QuadratureAdjoint(; autojacvec)  # Incompatible with ZygoteVJP for out-of-place problems
    end
end
