using KernelAbstractions
using CUDA

@kernel function metropolis_kernel!(samples, N, J, temperature, max_temp, cooling_steps)
    # Get thread index
    idx = @index(Global)
    
    # Initialize state for this thread
    current_state = rand([-1, 1], N)
    current_energy = hopfield_energy(current_state, J)
    
    # Annealing schedule from max_temp down to target temperature
    for step in 1:cooling_steps
        # Calculate current temperature
        current_temp = max_temp * (temperature/max_temp)^(step/cooling_steps)
        
        # Propose new state by flipping random spin
        flip_idx = rand(1:N)
        proposed_state = copy(current_state)
        proposed_state[flip_idx] *= -1
        proposed_energy = hopfield_energy(proposed_state, J)
        
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
    samples[idx] = copy(current_state)
end

function sample_boltzmann_KA_CPU(N::Int, J::Matrix{Float64}, temperature::Float64; num_samples::Int=1000, burn_in::Int=100, max_temp::Float64=100.0, cooling_steps::Int=1000)
    # Initialize storage for samples
    samples = [zeros(Int, N) for _ in 1:num_samples]
    
    # Create kernel instance
    kernel = metropolis_kernel!(CPU(), 64)
    
    # Launch kernel
    kernel(samples, N, J, temperature, max_temp, cooling_steps, ndrange=num_samples)
    
    return samples
end

function sample_boltzmann_KA_GPU(N::Int, J::Matrix{Float64}, temperature::Float64; num_samples::Int=1000, burn_in::Int=100, max_temp::Float64=100.0, cooling_steps::Int=1000)
    # Initialize storage for samples
    samples = [zeros(Int, N) for _ in 1:num_samples]
    
    # Create kernel instance    
    kernel = metropolis_kernel!(CUDA(), 64)
    
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
states = sample_boltzmann_KA_GPU(N, coupling_matrix, T)
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
    size=(800,600)
)