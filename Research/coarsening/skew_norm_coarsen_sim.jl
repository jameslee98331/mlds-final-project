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
alphas = [10^2, 10^3, 10^4, 10^5, Inf]  # robustification params alpha to use
seps = 1:0.5:5
n_reps = 5  # number of times to run the simulation
# seps = ["6.0", "7.0", "8.0", "9.0", "10.0", "15.0", "25.0", "50.0", "100.0", "200.0", "1000.0", "2000.0", "5000.0", "10000.0", "1e+05", "1e+06"]

for alpha in alphas
    for (i_n, n) in enumerate(ns)
        for (i_sep, sep) in enumerate(seps)

            all_data = h5read("./data_inputs/skew_normal_mixtures_1d-sep=$sep.jld", "skew_norm_data")
            k_posteriors = zeros(t_max, n_reps)
    
            for rep in 1:n_reps
    
                Random.seed!(n + rep) # Reset RNG
                shuffled_data = shuffle(all_data)
                data = [shuffled_data[j]::Float64 for j in 1:n]

                zeta = (1 / n) / ((1 / n) + (1 / alpha))

                println("rep = $rep, n = $n, sep = $sep, alpha = $alpha, zeta=$zeta")    
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

            save_fullpath = "./comp_outputs/k_posteriors-alpha=$alpha-n=$n-sep=$sep-1d_skew_norm_mixtures.jld"
            save(save_fullpath, "k_posteriors", k_posteriors)
            println("Saved to $save_fullpath")
            println()

        end
    end
end
