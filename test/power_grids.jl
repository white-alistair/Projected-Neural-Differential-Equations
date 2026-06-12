@testitem "constraints_jacobian vector case CPU vs ForwardDiff" begin
    using Random, MLDataDevices, ForwardDiff

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        device = MLDataDevices.cpu_device()
        rng = Random.MersenneTwister(42)
        system = system_type{Float64}(device)

        # Generate random voltage state with appropriate dimension
        u = rand(rng, input_dim)
        t = 0.0

        # Compare analytic Jacobian vs ForwardDiff
        J_analytic = ProjectedNDEs.constraints_jacobian(u, t, system)
        J_autodiff = ForwardDiff.jacobian(u -> ProjectedNDEs.constraints(u, t, system), u)

        @test isapprox(J_analytic, J_autodiff, rtol=1e-6)
    end
end

@testitem "constraints_jacobian matrix case CPU vs ForwardDiff" begin
    using Random, MLDataDevices, ForwardDiff

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        device = MLDataDevices.cpu_device()
        rng = Random.MersenneTwister(42)
        system = system_type{Float64}(device)

        # Generate batch of random voltage states
        batch_size = 5
        U = rand(rng, input_dim, batch_size)
        t = 0.0

        # Compute batched Jacobian
        J_analytic = ProjectedNDEs.constraints_jacobian(U, t, system)

        # Compare with ForwardDiff for each sample in batch
        for i in 1:batch_size
            u_i = U[:, i]
            J_autodiff = ForwardDiff.jacobian(u -> ProjectedNDEs.constraints(u, t, system), u_i)
            @test isapprox(J_analytic[:, :, i], J_autodiff, rtol=1e-6)
        end
    end
end

@testitem "constraints_jacobian vector vs matrix consistency (CPU)" begin
    using Random, MLDataDevices

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        device = MLDataDevices.cpu_device()
        rng = Random.MersenneTwister(42)
        system = system_type{Float64}(device)

        # Single sample test
        u = rand(rng, input_dim)
        U = reshape(copy(u), input_dim, 1)
        t = 0.0

        # Vector case vs first slice of matrix case
        J_vector = ProjectedNDEs.constraints_jacobian(u, t, system)
        J_matrix = ProjectedNDEs.constraints_jacobian(U, t, system)

        @test isapprox(J_vector, J_matrix[:, :, 1], rtol=1e-10)
    end
end

@testitem "constraints_jacobian vector case GPU vs CPU" begin
    using Random, MLDataDevices, CUDA

    if !CUDA.functional()
        @warn "CUDA not available, skipping GPU vs CPU vector tests"
        return
    end

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        rng = Random.MersenneTwister(42)

        # Create CPU and GPU systems
        system_cpu = system_type{Float32}(MLDataDevices.cpu_device())
        system_gpu = system_type{Float32}(MLDataDevices.gpu_device())

        # Create CPU and GPU inputs
        u_cpu = rand(rng, Float32, input_dim)
        u_gpu = CuArray(u_cpu)
        t = 0.0

        # Compute Jacobians
        J_cpu = ProjectedNDEs.constraints_jacobian(u_cpu, t, system_cpu)
        J_gpu = ProjectedNDEs.constraints_jacobian(u_gpu, t, system_gpu)

        # Compare GPU vs CPU
        @test isapprox(Array(J_gpu), J_cpu, rtol=1e-6)
    end
end

@testitem "constraints_jacobian vector case GPU vs ForwardDiff" begin
    using Random, MLDataDevices, ForwardDiff, CUDA

    if !CUDA.functional()
        @warn "CUDA not available, skipping GPU vs autodiff vector tests"
        return
    end

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        rng = Random.MersenneTwister(42)

        # Create systems
        system_cpu = system_type{Float32}(MLDataDevices.cpu_device())
        system_gpu = system_type{Float32}(MLDataDevices.gpu_device())

        # Create inputs
        u_cpu = rand(rng, Float32, input_dim)
        u_gpu = CuArray(u_cpu)
        t = 0.0

        # Compute Jacobians
        J_analytic = ProjectedNDEs.constraints_jacobian(u_gpu, t, system_gpu)
        J_autodiff = ForwardDiff.jacobian(u -> ProjectedNDEs.constraints(u, t, system_cpu), u_cpu)

        # Compare analytic vs ForwardDiff
        @test isapprox(Array(J_analytic), J_autodiff, rtol=1e-6)
    end
end

@testitem "constraints_jacobian matrix case GPU vs CPU" begin
    using Random, MLDataDevices, CUDA

    if !CUDA.functional()
        @warn "CUDA not available, skipping GPU vs CPU matrix tests"
        return
    end

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        rng = Random.MersenneTwister(42)

        # Create CPU and GPU systems
        system_cpu = system_type{Float32}(MLDataDevices.cpu_device())
        system_gpu = system_type{Float32}(MLDataDevices.gpu_device())

        # Create batch data
        batch_size = 5
        U_cpu = rand(rng, Float32, input_dim, batch_size)
        U_gpu = CuArray(U_cpu)
        t = 0.0

        # Compute Jacobians
        J_cpu = ProjectedNDEs.constraints_jacobian(U_cpu, t, system_cpu)
        J_gpu = ProjectedNDEs.constraints_jacobian(U_gpu, t, system_gpu)

        # Compare GPU vs CPU
        @test isapprox(Array(J_gpu), J_cpu, rtol=1e-6)
    end
end

@testitem "constraints_jacobian matrix case GPU vs ForwardDiff" begin
    using Random, MLDataDevices, ForwardDiff, CUDA

    if !CUDA.functional()
        @warn "CUDA not available, skipping GPU vs autodiff matrix tests"
        return
    end

    SYSTEMS_TO_TEST = [(ProjectedNDEs.IEEE14Bus, 28)]

    for (system_type, input_dim) in SYSTEMS_TO_TEST
        rng = Random.MersenneTwister(42)

        # Create systems
        system_cpu = system_type{Float32}(MLDataDevices.cpu_device())

        # Create batch data
        batch_size = 5
        U_cpu = rand(rng, Float32, input_dim, batch_size)
        t = 0.0

        # Check against ForwardDiff for first sample
        J_autodiff = ForwardDiff.jacobian(
            u -> ProjectedNDEs.constraints(u, t, system_cpu),
            U_cpu[:, 1]
        )

        # Also compute GPU result for comparison
        system_gpu = system_type{Float32}(MLDataDevices.gpu_device())
        U_gpu = CuArray(U_cpu)
        J_analytic = ProjectedNDEs.constraints_jacobian(U_gpu, t, system_gpu)

        # Compare analytic vs ForwardDiff
        @test isapprox(Array(J_analytic)[:, :, 1], J_autodiff, rtol=1e-6)
    end
end
