using BayesianMixtures
using Distributions
using HDF5
using JLD
using StatsBase
using Random

# options
mcmc_its = 100000 # total number of MCMC sweeps to run
mcmc_burn = Int(mcmc_its / 10) # number of burn-in iterations
t_max = 127

ns = [250, 500, 1000, 2500, 5000, 10000]
n_reps = 5
seps = 1:0.5:5
n_seps = length(seps)

for (i_n, n) in enumerate(ns)
    for (i_sep, sep) in enumerate(seps)
        
        # Read data
        all_data = h5read("./data_inputs/gaussian_data-sep=$sep.jld", "gaussian_data")
        dt = fit(ZScoreTransform, all_data, dims=1)
        standardised_data = StatsBase.transform(dt, all_data)

        t_posteriors = zeros(t_max, n_reps)
        k_posteriors = zeros(t_max, n_reps)

        for rep in 1:n_reps

            Random.seed!(n + rep) # Reset RNG
            shuffled_data = shuffle(standardised_data)
            data = [shuffled_data[j, :]::Array{Float64} for j in 1:n]

            # MFM with univariate Normal components
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
            println("n = $n, sep = $sep, rep=$rep")
            dpm_result = BayesianMixtures.run_sampler(dpm_options)
            mfm_result = BayesianMixtures.run_sampler(mfm_options) # Results of Miller and Harrison (2018)

            # Get the posterior on k (number of components) 
            t_posterior = BayesianMixtures.t_posterior(dpm_result)
            t_posteriors[:, rep] = t_posterior

            # Get the posterior on k (number of components) 
            k_posterior = BayesianMixtures.k_posterior(mfm_result)
            k_posteriors[:, rep] = k_posterior
        end

        save(
            "./comp_outputs/t_posteriors-mvn-dpm-sep=$sep-n=$n-gaussian_mixtures.jld",
            "t_posteriors",
            t_posteriors
        )

        save(
            "./comp_outputs/k_posteriors-mvn-mfm-sep=$sep-n=$n-gaussian_mixtures.jld",
            "k_posteriors",
            k_posteriors
        )

        println("Saved posteriors for n=$n, sep=$sep")
        println()
    end
end
