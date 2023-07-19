using BayesianMixtures
using Distributions
using HDF5
using JLD
using Random

# Simulate some data
mean1 = [0, 0]
C = [1 0; 0 1]
d1 = MvNormal(mean1, C)
p1 = 0.5

# options
mcmc_its = 10^5 # total number of MCMC sweeps to run
mcmc_burn = Int(mcmc_its / 10) # number of burn-in iterations
t_max = 100

ns = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
n_reps = 5
all_data = h5read("../../data/gaussian_data.jld", "gaussian_data")
data_name = "gaussian_mixtures"

for (i_n, n) in enumerate(ns)

    t_posteriors = zeros(t_max, n_reps)
    k_posteriors = zeros(t_max, n_reps)

    for rep in 1:n_reps

        # shuflle dataset and extract a subset of length n
        shuffled_data = shuffle(all_data)
        data = [shuffled_data[j, :]::Array{Float64} for j in 1:n]

        save(
            "./input_data/raw-data-mvn-dpm-n=$n-$data_name.jld",
            "data",
            data
        )

        # setup sampler options
        dpm_options = BayesianMixtures.options(
            "MVN",
            "DPM",
            data,
            mcmc_its,
            n_burn=mcmc_burn,
            t_max=t_max
        )
        mfm_options = BayesianMixtures.options(
            "MVN",
            "MFM",
            data,
            mcmc_its,
            n_burn=mcmc_burn,
            t_max=t_max
        )

        # Run MCMC sampler - Jain-Neal split-merge samplers
        println("n = $n, rep=$rep")
        dpm_result = BayesianMixtures.run_sampler(dpm_options)
        mfm_result = BayesianMixtures.run_sampler(mfm_options) # Results of Miller and Harrison (2018)

        # Get the posterior on t (number of clusters - non-empty components) 
        t_posterior = BayesianMixtures.t_posterior(dpm_result)
        t_posteriors[:, rep] = t_posterior

        # Get the posterior on k (number of components) 
        k_posterior = BayesianMixtures.k_posterior(mfm_result)
        k_posteriors[:, rep] = k_posterior
    end

    save(
        "./comp_outputs/t_posteriors-mvn-dpm-n=$n-$data_name.jld",
        "t_posteriors",
        t_posteriors
    )

    save(
        "./comp_outputs/k_posteriors-mvn-mfm-n=$n-$data_name.jld",
        "k_posteriors",
        k_posteriors
    )

    println("Saved t_posteriors for base case")
    println()
end
