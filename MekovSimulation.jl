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

    # Initialize noise
    rng = Xoshiro128Star(1234)
    dW = sqrt(dt) * randn(rng, 2, nt - 1)

    # Initialize feedback integral
    I_hist = zeros(ComplexF64, nt)

    for i in 1:(nt-1)
        # Compute deterministic step using RK4
        k1 = dt * mekov!(similar(u[i]), u[i], p, t[i])
        k2 = dt * mekov!(similar(u[i]), u[i] + k1 / 2, p, t[i] + dt / 2)
        k3 = dt * mekov!(similar(u[i]), u[i] + k2 / 2, p, t[i] + dt / 2)
        k4 = dt * mekov!(similar(u[i]), u[i] + k3, p, t[i] + dt)

        # Add noise term
        noise_term = σ_mekov!(similar(u[i]), u[i], p, t[i])

        # Update state
        u[i+1] = u[i] + (k1 + 2k2 + 2k3 + k4) / 6 + noise_term * [dW[1, i]; dW[2, i]]

        # Compute measurement signal
        I_hist[i] = sqrt(2 * κ) * (u[i][1] + conj(u[i][1])) - dW[1, i] / sqrt(2 * κ)
    end

    # Compute feedback signal using convolution with kernel
    I = zeros(ComplexF64, nt)
    for i in 1:nt
        for j in 1:i
            I[i] += I_hist[j] * kernel_mekov!(t[i] - t[j], t_0=t_0, s=s) * dt
        end
    end

    return t, u, I
end


# dSx/dt = -ωzSy
# dSy/dt = ωzSx - 2/√N(λ(t)a + conj(λ(t))conj(a))Sz
# dSz/dt = 2/√N(λ(t)a + conj(λ(t))conj(a))Sy


