using QuantumOptics, DifferentialEquations
include("HenryLib.jl")

fb, sb, bases, a, Sx, Sy, Sz, idOp = make_operators(fockmax, Nspin)

