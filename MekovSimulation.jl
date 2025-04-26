using DifferentialEquations, LaTeXStrings, JLD2, Printf, PyPlot, LsqFit, RandomNumbers.Xorshifts, DiffEqGPU, CUDA, ProgressLogging, Profile, StaticArrays
include("HenryLib.jl")


a = 1
b = 2

function mekov!(du, u, p, t)
    Δ, ω_z, κ, g, G, I = p
    du[a] = (-1im * Δ - κ) * u[a] - 2im * g * (u[b] + conj(u[b]))
    du[b] = (-1im * ω_z) * u[b] - 2im * g * (u[a] + conj(u[a])) + G * I
end

function σ_mekov!(du, u, p, t)
    Δ, ω_z, κ, g, G, I = p
    du[a] = sqrt(κ / 2)
    du[b] = 0
end

function kernel_mekov!(t; t_0=1, s=2)
    return (t_0 / (t + t_0))^s
end

function evolve_mekov(u₀, tspan, p, dt; kernel_params=(1.0, 2.0))
    # Unpack parameters
    Δ, ω_z, κ, g, G, _ = p
    t_0, s = kernel_params

    # Initialize arrays
    t = tspan[1]:dt:tspan[2]
    nt = length(t)
    u = Vector{typeof(u₀)}(undef, nt)
    u[1] = u₀

    # Create SDEProblem
    prob = SDEProblem(mekov!, σ_mekov!, u₀, tspan, p)

    # Initialize integrator with fixed timestep
    integrator = init(prob, SRIW1(); dt=dt, adaptive=false)

    # Initialize arrays for measurement and feedback
    I_hist = zeros(ComplexF64, nt)

    for i in 1:(nt-1)
        # Take a step with the integrator
        step!(integrator)
        
        # Store the state
        u[i+1] = integrator.u

        # Compute measurement signal
        I_hist[i] = sqrt(2 * κ) * (integrator.u[1] + conj(integrator.u[1])) - integrator.W[1] / sqrt(2 * κ)
        for j in 1:i
            I[i] += I_hist[j] * kernel_mekov!(t[i] - t[j], t_0=t_0, s=s) * dt
        end
        integrator.p[6] = I[i]
    end

    # Compute feedback signal using convolution with kernel
    I = zeros(ComplexF64, nt)
    for i in 1:nt
        
    end

    return t, u, I
end


# dSx/dt = -ωzSy
# dSy/dt = ωzSx - 2/√N(λ(t)a + conj(λ(t))conj(a))Sz
# dSz/dt = 2/√N(λ(t)a + conj(λ(t))conj(a))Sy

# Run simulation
tspan = (0.0, 10.0)
dt = 0.01
u₀ = [0.0 + 0.0im, 0.0 + 0.0im]  # Initial conditions
p = [1.0, 1.0, 0.1, 0.5, 1.0, 0.0]  # [Δ, ω_z, κ, g, G, I]

t, u, I = evolve_mekov(u₀, tspan, p, dt)

# Extract real parts
a_real = real.(getindex.(u, 1))
b_real = real.(getindex.(u, 2))

# Create plots
using Plots
pyplot()

p1 = plot(t, a_real, label="Re(a)", xlabel="Time", ylabel="Value", title="Real part of a")
p2 = plot(t, b_real, label="Re(b)", xlabel="Time", ylabel="Value", title="Real part of b")

# Combine plots
plot(p1, p2, layout=(2,1), size=(800,600))
savefig("mekov_simulation.png")

# Also plot the feedback signal
p3 = plot(t, real.(I), label="Re(I)", xlabel="Time", ylabel="Value", title="Real part of feedback signal")
savefig("feedback_signal.png")


