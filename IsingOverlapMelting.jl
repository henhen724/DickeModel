function generate_symmetric_matrix(n::Int)
    # Initialize an nxn matrix with random values
    matrix = randn(n,n)
    
    # Make it symmetric by averaging with its transpose
    symmetric_matrix = (matrix + matrix') / 2
    
    return symmetric_matrix
end

function hopfield_energy(state::Vector{<:Number}, J::Matrix{Float64})
    # Calculate the Hopfield energy for a given state and coupling matrix
    # E = -1/2 ∑ᵢⱼ Jᵢⱼsᵢsⱼ
    N = length(state)
    energy = 0.0
    
    for i in 1:N
        for j in 1:N
            energy -= 0.5 * J[i,j] * state[i] * state[j]
        end
    end
    
    return energy
end


function count_local_minima(N::Int)
    # Generate random coupling matrix
    J = generate_symmetric_matrix(N)
    
    # Initialize counters
    local_minima_count = 0
    total_states = 2^N
    
    # Check all possible states
    for state_idx in 0:(total_states-1)
        # Convert integer to spin state vector {-1,1}^N
        state = [((state_idx >> i) & 1) * 2 - 1 for i in 0:(N-1)]
        
        # Check if current state is local minimum by flipping each spin
        is_minimum = true
        current_energy = hopfield_energy(state, J)
        
        for i in 1:N
            # Create copy and flip i-th spin
            flipped_state = copy(state)
            flipped_state[i] *= -1
            
            # Compare energies
            if hopfield_energy(flipped_state, J) < current_energy
                is_minimum = false
                break
            end
        end
        
        if is_minimum
            local_minima_count += 1
        end
    end
    
    return local_minima_count, J
end

function find_local_minima(N::Int, J::Matrix{Float64})
    # Initialize list to store local minima states
    local_minima_states = Vector{Vector{Int}}()
    total_states = 2^N
    
    # Check all possible states
    for state_idx in 0:(total_states-1)
        # Convert integer to spin state vector {-1,1}^N
        state = [((state_idx >> i) & 1) * 2 - 1 for i in 0:(N-1)]
        
        # Check if current state is local minimum by flipping each spin
        is_minimum = true
        current_energy = hopfield_energy(state, J)
        
        for i in 1:N
            # Create copy and flip i-th spin
            flipped_state = copy(state)
            flipped_state[i] *= -1
            
            # Compare energies
            if hopfield_energy(flipped_state, J) < current_energy
                is_minimum = false
                break
            end
        end
        
        if is_minimum
            push!(local_minima_states, state)
        end
    end
    
    return local_minima_states
end

function sample_boltzmann(N::Int, J::Matrix{Float64}, temperature::Float64; num_samples::Int=1000, burn_in::Int=100, max_temp::Float64=100.0, cooling_steps::Int=1000)
    # Initialize storage for samples
    samples = Vector{Vector{Int}}()
    
    for sample in 1:num_samples
        # Start from random state at high temperature
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
        
        # Store the final state after annealing
        push!(samples, copy(current_state))
    end
    
    return samples
end


function calculate_overlap_histogram(states::Vector{Vector{Int}})
    # Calculate number of states and their length
    num_states = length(states)
    if num_states == 0
        return Dict{Float64,Int}()
    end
    N = length(states[1])
    
    # Calculate all pairwise overlaps
    overlaps = Float64[]
    for i in 1:num_states
        for j in 1:num_states
            # Calculate normalized overlap between states i and j
            overlap = sum(states[i] .* states[j]) / N
            push!(overlaps, overlap)
        end
    end
    
    # Create histogram by counting occurrences of each overlap value
    histogram = Dict{Float64,Int}()
    for overlap in overlaps
        histogram[overlap] = get(histogram, overlap, 0) + 1
    end
    
    return histogram
end

using Plots
gr()  # Use GR backend which works well in consolefind_local_minima(N, coupling_matrix)

# Example usage:
N = 20  # Small system size for demonstration
# Generate coupling matrix using helper function
coupling_matrix = generate_symmetric_matrix(N)
T = 30.0
states = sample_boltzmann(N, coupling_matrix, T)
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

# Save the plot to a file
savefig(p, "overlap_histogram.png")

# Print some information about the results
println("Number of local minima found: ", length(minima))
println("Overlap histogram:")
for (overlap, count) in overlap_histogram
    println("Overlap = $overlap: $count pairs")
end
