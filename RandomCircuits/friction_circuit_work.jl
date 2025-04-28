using Random;
Random.seed!(1);
using QuantumClifford, QuantumCliffordPlots, Plots;
plot(random_stabilizer(20, 30), xzcomponents=:split);
savefig("plot-randstab.png");
nothing;

