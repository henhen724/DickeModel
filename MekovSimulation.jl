using DifferentialEquations, LaTeXStrings, JLD2, Printf, PyPlot, LsqFit, RandomNumbers.Xorshifts, DiffEqGPU, CUDA, ProgressLogging, Profile, StaticArrays
include("HenryLib.jl")


a = 1
b = 2

function mekov!(du, u, p, t)
    Δ, ω_z, κ, λ, N = p
    du[a] = (-1im * Δ - κ) * u[a] - 2im * λ(t) * (u[b] + conj(u[b])) / sqrt(N)
    du[b] = (-1im * ω_z) * u[b] - 2im * λ(t) * (u[a] + conj(u[a])) / sqrt(N)
end

function σ_mekov!(du, u, p, t)
    Δ, κ, λ, N = p
    du[a] = sqrt(κ/2)
    du[b] = 0
end



# dSx/dt = -ωzSy
# dSy/dt = ωzSx - 2/√N(λ(t)a + conj(λ(t))conj(a))Sz
# dSz/dt = 2/√N(λ(t)a + conj(λ(t))conj(a))Sy


