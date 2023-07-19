module Simulation
using Distributions
using PyPlot
using HDF5
using JLD
using Random

include("setup.jl")

# Settings
ns = [100, 250, 500, 1000, 2500, 5000, 10000] # sample sizes n to use
alphas = [Inf, 10^5, 10^4, 10^3, 10^2, 10]  # robustification params alpha to use
n_reps = 5  # number of times to run the simulation
mcmc_its = 10^5  # number of MCMC iterations
mcmc_burn = Int(mcmc_its / 10)  # number of iterations to discard as burn-in
t_max = 10

for alpha in alphas
    for (i_n, n) in enumerate(ns)

        k_posteriors = zeros(t_max, n_reps)

        for rep in 1:n_reps
            Random.seed!(n + rep) # Reset RNG
            data = [
                (rand() < p1 ? skewrnd(1, l1, s1, a1) : skewrnd(1, l2, s2, a2))[1]::Float64
                for j = 1:n
            ] # Sample data

            zeta = (1 / n) / ((1 / n) + (1 / alpha))

            println("n = $n, rep = $rep, alpha = $alpha, zeta=$zeta")

            # Run sampler
            elapsed_time = (
                @elapsed p, theta, k_r, v_r, art, arv, m_r, s_r = sampler(
                data, mcmc_its, t_max, c, sigma, zeta
            )
            )

            time_per_step = elapsed_time / mcmc_its
            println("Elapsed time = $elapsed_time seconds")
            println("Time per step = $time_per_step seconds")

            # Compute posterior on k
            counts, bins = hist(k_r[mcmc_burn+1:end], range(1, t_max + 1, t_max + 1))
            k_posteriors[:, rep] = counts / (mcmc_its - mcmc_burn)
        end

        save_filename = "k_posteriors-alpha=$alpha-n=$n.jld"
        save_fullpath = "./comp_outputs/" * save_filename
        save(save_fullpath, "k_posteriors", k_posteriors)
        println("Saved to $save_fullpath")
        println()
    end
end

end # module