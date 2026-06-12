function get_mlp((input_size, output_size)::Pair, hidden_layers, hidden_width, activation; use_skip=false)
    layers = []

    # First hidden layer (no skip - dimension mismatch)
    push!(layers, Lux.Dense(input_size => hidden_width, activation))

    # Remaining hidden layers
    for i in 1:(hidden_layers - 1)
        if use_skip
            # Pre-activation residual: output = x + f(x)
            push!(layers, Lux.SkipConnection(Lux.Dense(hidden_width => hidden_width, activation), +))
        else
            push!(layers, Lux.Dense(hidden_width => hidden_width, activation))
        end
    end

    # Output layer (no skip - dimension mismatch)
    push!(layers, Lux.Dense(hidden_width => output_size))

    return Lux.Chain(layers)
end
