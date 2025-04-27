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

function calculate_binder_cumulant(overlap_histogram::Dict{Float64,<:Number})
    if isempty(overlap_histogram)
        return 0.0
    end
    
    # Calculate moments directly using the probability distribution
    m2 = sum(overlap^2 * prob for (overlap, prob) in overlap_histogram)
    m4 = sum(overlap^4 * prob for (overlap, prob) in overlap_histogram)
    
    # Calculate Binder cumulant
    binder_cumulant = 1 - (m4 / (3 * m2^2))
    
    return binder_cumulant
end

function calculate_averaged_overlap_histogram(N::Int, num_matrices::Int, T::Float64)
    # Initialize empty dictionary to store the summed histograms
    summed_histogram = Dict{Float64, Float64}()
    
    for i in 1:num_matrices
        # Generate a new coupling matrix
        coupling_matrix = generate_symmetric_matrix(N)
        
        # Sample states using Boltzmann sampling
        states = sample_boltzmann(N, coupling_matrix, T)
        
        # Calculate overlap histogram for this matrix
        overlap_hist = calculate_overlap_histogram(states)
        
        # Get total count for normalization
        total_count = sum(values(overlap_hist))
        
        # Add normalized counts to summed histogram
        for (overlap, count) in overlap_hist
            normalized_count = count / total_count
            if haskey(summed_histogram, overlap)
                summed_histogram[overlap] += normalized_count
            else
                summed_histogram[overlap] = normalized_count
            end
        end
    end
    
    # Calculate average by dividing by number of matrices
    averaged_histogram = Dict{Float64, Float64}()
    for (overlap, sum_norm_count) in summed_histogram
        averaged_histogram[overlap] = sum_norm_count / num_matrices
    end
    
    return averaged_histogram
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

function sample_with_atom_number_shot_noise(coupling_matrix::Matrix{Float64}, T::Float64, fractional_atom_number_variance::Float64;  num_samples::Int=1000, burn_in::Int=100, max_temp::Float64=100.0, cooling_steps::Int=1000)
    N = size(coupling_matrix, 1)
    states = Vector{Vector{Int}}()
    
    for _ in 1:num_samples
        # Generate random atom number fluctuations
        mean_atom_number = 1.0
        atom_numbers = (1.0 .+ fractional_atom_number_variance*randn(N)).*ones(N)*mean_atom_number
        
        # Clip negative atom numbers to zero
        atom_numbers = max.(atom_numbers, 0.0)

        # Apply atom number fluctuations to coupling matrix
        fluctuated_coupling = Diagonal(atom_numbers) * coupling_matrix * Diagonal(atom_numbers)
        
        # Start from random state at high temperature
        state = rand([-1, 1], N)
        current_energy = hopfield_energy(state, fluctuated_coupling)
        
        # Annealing schedule from max_temp down to target temperature
        for step in 1:cooling_steps
            # Calculate current temperature
            current_temp = max_temp * (T/max_temp)^(step/cooling_steps)
            
            # Metropolis steps at current temperature
            for _ in 1:N
                # Propose spin flip
                flip_idx = rand(1:N)
                state_copy = copy(state)
                state_copy[flip_idx] *= -1
                
                # Calculate energy change
                proposed_energy = hopfield_energy(state_copy, fluctuated_coupling)
                delta_E = proposed_energy - current_energy
                
                # Metropolis acceptance criterion
                if delta_E <= 0 || rand() < exp(-delta_E / current_temp)
                    state = state_copy
                    current_energy = proposed_energy
                end
            end
        end
        
        push!(states, copy(state))
    end
    
    return states
end