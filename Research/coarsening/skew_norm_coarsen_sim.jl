module Simulation
using Distributions
using PyPlot
using HDF5
using JLD
using Random

include("setup.jl")

# Settings
mcmc_its = 10^5  # number of MCMC iterations
mcmc_burn = Int(mcmc_its / 10)  # number of iterations to discard as burn-in
t_max = 10

ns = [100, 250, 500, 1000, 2500, 5000, 10000] # sample sizes n to use
alphas = [10, 10^2, 10^3, 10^4, 10^5, 10^6, Inf]  # robustification params alpha to use
n_reps = 5  # number of times to run the simulation

all_data = h5read("./data_inputs/skew_norm_data.jld", "skew_norm_data")
data_name = "skew_norm_mixtures"

for alpha in alphas
    for (i_n, n) in enumerate(ns)

        k_posteriors = zeros(t_max, n_reps)

        for rep in 1:n_reps
            Random.seed!(n + rep) # Reset RNG
            shuffled_data = shuffle(all_data)
            
            # TODO: update setup.jl and core.jl to work with MV data
            data = [shuffled_data[j, 1]::Float64 for j in 1:n]
            save(
                "./data_inputs/raw-data-n=$n-$data_name.jld",
                "data",
                data
            )

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

        save_fullpath = "./comp_outputs/k_posteriors-alpha=$alpha-n=$n-$data_name.jld"
        save(save_fullpath, "k_posteriors", k_posteriors)
        println("Saved to $save_fullpath")
        println()
    end
end

end # module