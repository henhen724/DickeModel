using Plots, Statistics
gr()  # Use GR backend which works well in console

# Example usage:
N = 20  # Small system size for demonstration
# Generate coupling matrix using helper function
coupling_matrix = generate_symmetric_matrix(N)
T = 0.1
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


averaged_histogram = calculate_averaged_overlap_histogram(N, 10, 0.1)

# Plot the averaged overlap histogram
overlaps = collect(keys(averaged_histogram))
counts = collect(values(averaged_histogram))

p = bar(overlaps, counts,
    xlabel="Overlap",
    ylabel="Count",
    title="Averaged Overlap Histogram \$N=$N\$ \$T = $T\\tilde{J}\$",
    legend=false,
    grid=true,
    size=(800,600)
)

# Save the plot to a file
savefig(p, "averaged_overlap_histogram.png")

# Calculate Binder cumulant for our averaged histogram
binder_cumulant = calculate_binder_cumulant(overlap_histogram)
println("Binder cumulant at T = $T: ", binder_cumulant)




using LinearAlgebra

# Parameters
N = 20  # Number of spins
T = 0.1  # Temperature
σ²_values = LinRange(0.0, 1.0, 6)  # Different atom number variances to test

# Generate a single J matrix
J = generate_symmetric_matrix(N)

# Create array of plots
plots = []
for (i, σ²) in enumerate(σ²_values)
    # Sample states with atom number shot noise
    states = sample_with_atom_number_shot_noise(J, T, σ²)
    
    # Calculate overlap histogram
    histogram = calculate_overlap_histogram(states)
    total_count = sum(values(histogram))
    histogram = Dict(k => v/total_count for (k,v) in histogram)
    
    # Extract data for plotting
    overlaps = collect(keys(histogram))
    counts = collect(values(histogram))
    
    # Create plot
    p = bar(overlaps, counts,
        title="σ² = $σ²",
        xlabel="Overlap",
        ylabel="Count",
        legend=false,
        grid=true
    )
    push!(plots, p)
end 

using Measures

# Combine plots into 2x3 array
final_plot = plot(plots..., layout=(2,3), size=(1200,800), left_margin=4mm, dpi=300, ylims=(0, 0.35))
savefig(final_plot, "overlap_histograms_T$(T)_N$(N).png")


# Parameters
N = 20  # Number of spins
T = 0.1  # Temperature
σ²_values = LinRange(0.0, 1.0, 6)  # Different atom number variances to test

# Generate a single J matrix
J_matrices = [generate_symmetric_matrix(N) for _ in 1:20]

# Create array of plots
plots = []
for (i, σ²) in enumerate(σ²_values)
    # Initialize empty dictionary for averaged histogram
    averaged_histogram = Dict{Float64, Float64}()
    
    # Sample and average over all J matrices
    for J in J_matrices
        # Sample states with atom number shot noise
        states = sample_with_atom_number_shot_noise(J, T, σ²)
        
        # Calculate overlap histogram for this J matrix
        histogram = calculate_overlap_histogram(states)
        
        # Add normalized counts to averaged histogram
        total_count = sum(values(histogram))
        for (overlap, count) in histogram
            norm_count = count / total_count
            averaged_histogram[overlap] = get(averaged_histogram, overlap, 0.0) + norm_count
        end
    end
    
    # Normalize by number of matrices
    for overlap in keys(averaged_histogram)
        averaged_histogram[overlap] /= length(J_matrices)
    end
    
    # Extract data for plotting
    overlaps = collect(keys(averaged_histogram))
    counts = collect(values(averaged_histogram))
    
    # Create plot
    p = bar(overlaps, counts,
        title="σ² = $σ²",
        xlabel="Overlap",
        ylabel="Count",
        legend=false,
        grid=true
    )
    push!(plots, p)
end 

# Find global y-axis limits
# Update plots with consistent y-axis
final_plot = plot(plots..., layout=(2,3), size=(1200,800), left_margin=4mm, dpi=300, ylims=(0, 0.4))
savefig(final_plot, "averaged_overlap_histograms_T$(T)_N$(N).png")

# Print summary information
println("Simulation Summary (T = $T):")
for (i, σ²) in enumerate(σ²_values)
    states = sample_with_atom_number_shot_noise(J, T, σ²)
    histogram = calculate_overlap_histogram(states)
    println("σ² = $σ²: Sampled $(length(states)) states")
    println("Overlap distribution:")
    for (overlap, count) in histogram
        println("  Overlap = $overlap: $count pairs")
    end
end

# Create temperature schedule for annealing visualization
max_temp = 100.0
cooling_steps = 100
rounds = 3
step = 1:3*cooling_steps

T_schedule = max_temp * (T/max_temp).^((step .% cooling_steps)./cooling_steps)

# Create array to mark where new atom numbers are sampled
# Assuming we sample new atom numbers every 10 steps (adjust as needed)
sampling_points = 1:cooling_steps:3*cooling_steps

# Create annealing schedule plot
p_schedule = plot(step, T_schedule,
    title="Annealing Schedule",
    xlabel="Annealing Step",
    ylabel="Temperature",
    label="Temperature",
    linewidth=2,
    grid=true, dpi=300
)

vline!(sampling_points .-3 .+ cooling_steps, 
           label="Sample State", 
           color=:red, 
           linestyle=:dash,
           alpha=2.0)
vline!(sampling_points.+ 1, 
           label="Draw New Atom Numbers", 
           color=:green, 
           linestyle=:dash,
           alpha=2.0)

# Save the annealing schedule plot
savefig(p_schedule, "annealing_schedule_N$(N).png")


# Calculate Binder cumulants across temperature and disorder ranges
include("IsingOverlapMeltingLib.jl")

T_range = range(0.0, 3.0, length=20)
σ²_range = range(0.0, 1.0, length=20)
binder_cumulants = zeros(length(T_range), length(σ²_range))

N=20

J_matrices = [generate_symmetric_matrix(N) for _ in 1:20]

using ProgressMeter, LinearAlgebra, Plots


@showprogress "Temperature scan: " for (i, T_scan) in enumerate(T_range)
    @showprogress "σ² scan: " for (j, σ²_scan) in enumerate(σ²_range)
        # Sample states for this T, σ² combination using multiple threads
        histograms = Vector{Dict{Float64,Int}}(undef, length(J_matrices))
        @showprogress "Threads.@threads for k in eachindex(J_matrices)" for k in eachindex(J_matrices)
            histograms[k] = calculate_overlap_histogram(sample_with_atom_number_shot_noise(J_matrices[k], T_scan, σ²_scan))
        end

        # Combine histograms
        total_count = sum(sum(values(h)) for h in histograms)
        averaged_histogram = Dict{Float64, Float64}()
        for histogram in histograms
            for (overlap, count) in histogram
                norm_count = count / total_count
                averaged_histogram[overlap] = get(averaged_histogram, overlap, 0.0) + norm_count
            end
        end

        # Calculate Binder cumulant: 1 - m4/(3*m2^2)
        binder_cumulants[i,j] = calculate_binder_cumulant(averaged_histogram)
    end
end

# Create heatmap of Binder cumulants
p_binder = heatmap(T_range, σ²_range, binder_cumulants',
    title="Binder Cumulant",
    xlabel="Temperature",
    ylabel="σ²",
    color=:viridis,
    dpi=300
)

# Save the Binder cumulant plot
savefig(p_binder, "binder_cumulant_N$(N).png")
