using KernelAbstractions
using CUDA

function hopfield_energy_kernel!(energy, state, J)
    energy = 0.0
    @inbounds for i in 1:size(states, 2)
        @inbounds for j in 1:size(states, 2)
            energy -= 0.5 * state[i] * J[i, j] * state[j]
        end
    end

    return energy
end


@kernel function annealed_metropolis_kernel(samples, @Const(N), @Const(J), @Const(temperature), @Const(max_temp), @Const(cooling_steps))
    # Get thread index
    idx = @index(Global)

    # Initialize state for this thread
    current_state = rand([-1, 1], N)
    current_energy = cal

    # Annealing schedule from max_temp down to target temperature
    for step in 1:cooling_steps
        # Calculate current temperature
        current_temp = max_temp * (temperature / max_temp)^(step / cooling_steps)

        # Propose new state by flipping random spin
        flip_idx = rand(1:N)
        proposed_state = copy(current_state)
        proposed_state[flip_idx] *= -1
        proposed_energy = -0.5 * proposed_state' * J * proposed_state

        # Calculate acceptance probability
        delta_E = proposed_energy - current_energy
        acceptance_prob = exp(-delta_E / current_temp)

        # Accept or reject move
        if rand() < acceptance_prob
            current_state = proposed_state
            current_energy = proposed_energy
        end
    end

    # Store final state in samples array
    samples[idx, :] = copy(current_state)
end

function sample_boltzmann_KA_CPU(N::Int, J::Matrix{Float64}, temperature::Float64; num_samples::Int=1000, max_temp::Float64=100.0, cooling_steps::Int=1000)
    # Initialize storage for samples
    samples = zeros(Int, (num_samples, N))

    # Create kernel instance
    kernel = annealed_metropolis_kernel(CPU(), 64)

    # Launch kernel
    kernel(samples, N, J, temperature, max_temp, cooling_steps, ndrange=num_samples)

    return samples
end

using Plots, Statistics
gr()  # Use GR backend which works well in console
include("IsingOverlapMeltingLib.jl")

# Example usage:
N = 20  # Small system size for demonstration
# Generate coupling matrix using helper function
coupling_matrix = generate_symmetric_matrix(N)
T = 0.1
states = sample_boltzmann_KA_CPU(N, coupling_matrix, T)
overlap_histogram = calculate_overlap_histogram(states)

# Plot the overlap histogram
overlaps = collect(keys(overlap_histogram))
counts = collect(values(overlap_histogram))

# Create and save the plot
p = bar(overlaps, counts,
    xlabel="Overlap",
    ylabel="Count",
    title="Overlap Histogram \$N=$N\$ \$T = $T\\tilde{J}\$",
    legend=false,
    grid=true,
    size=(800, 600)
)

using BenchmarkTools

# Benchmark parameters
N_bench = 100  # System size for benchmarking
J_bench = generate_symmetric_matrix(N_bench)
T_bench = 0.1
num_samples_bench = 1000

println("Benchmarking three methods for sampling Boltzmann distribution")
println("Parameters: N=$N_bench, T=$T_bench, num_samples=$num_samples_bench")
println("\nNaive CPU Implementation:")
@btime sample_boltzmann($N_bench, $J_bench, $T_bench, num_samples=$num_samples_bench)

println("\nKernelAbstractions CPU Implementation:")
@btime sample_boltzmann_KA_CPU($N_bench, $J_bench, $T_bench, num_samples=$num_samples_bench)

println("\nKernelAbstractions GPU Implementation:")
@btime sample_boltzmann_KA_GPU($N_bench, $J_bench, $T_bench, num_samples=$num_samples_bench)
