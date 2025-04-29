using ProgressMeter, LinearAlgebra, Plots, Statistics
include("IsingOverlapMeltingLib.jl")

T_range = range(0.0, 3.0, length=20)
σ²_range = [0.0]
binder_cumulants = zeros(length(T_range), length(σ²_range))

N=20

J_matrices = [generate_symmetric_matrix(N) for _ in 1:5]


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
    title="Phase Diagram (Binder Cumulant)",
    xlabel="Temperature",
    ylabel="σ²",
    color=:viridis,
    clims=(0.0, 0.75),
    dpi=300
)

# Save the Binder cumulant plot
savefig(p_binder, "binder_cumulant_N$(N).png")