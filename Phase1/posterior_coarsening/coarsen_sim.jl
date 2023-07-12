module Simulation

using HDF5
using JLD
using PyPlot

include("./coarsen_setup.jl")

# Settings
# ns = [20, 100, 500, 2000, 10000] #, 100000]  # sample sizes n to use
ns = [100, 250, 500, 1000, 2500, 5000, 10000]
alphas = [Inf, 10000, 1000, 100]  # robustification params alpha to use
n_reps = 5  # number of times to run the simulation
mcmc_its = 100000  # number of MCMC iterations
mcmc_burn = Int(mcmc_its / 10)  # number of iterations to discard as burn-in
t_max = 10

for (i_a, alpha) in enumerate(alphas)
    for (i_n, n) in enumerate(ns)

        k_posteriors = zeros(t_max, n_reps)

        for rep in 1:n_reps
            Random.seed!(n + rep) # Reset RNG
            data = [
                (rand() < p1 ? skewrnd(1, l1, s1, a1) : skewrnd(1, l2, s2, a2))[1]::Float64
                for j = 1:n
            ] # Sample data

            # Run sampler
            zeta = alpha / (n + alpha)

            elapsed_time = (
                @elapsed p, theta, k_r, v_r, art, arv, m_r, s_r = sampler(
                data, mcmc_its, t_max, c, sigma, zeta
            )
            )
            time_per_step = elapsed_time / mcmc_its
            println("n = $n, rep = $rep, alpha = $alpha")
            println("Elapsed time = $elapsed_time seconds")
            println("Time per step = $time_per_step seconds")
            println()

            # Compute posterior on k
            counts, bins = hist(k_r[mcmc_burn+1:end], range(0, t_max, t_max + 1))
            k_posteriors[:, rep] = counts / (mcmc_its - mcmc_burn)
        end
        save("../comp_outputs/k_posteriors-alpha=$alpha-n=$n.jld", "k_posteriors", k_posteriors)
    end
end

end # module