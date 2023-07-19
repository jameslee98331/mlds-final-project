using BayesianMixtures
using Distributions
using HDF5
using JLD

# Simulate some data
mean1 = [0, 0]
C = [1 0; 0 1]
d1 = MvNormal(mean1, C)
p1 = 0.5

# options
mcmc_its = 100000 # total number of MCMC sweeps to run
mcmc_burn = Int(mcmc_its / 10) # number of burn-in iterations
t_max = 100

ns = [100, 250, 500, 1000, 2500, 5000, 10000]
n_reps = 5
seps = 1:0.25:5
n_seps = length(seps)

for (i_n, n) in enumerate(ns)
    for (i_sep, sep) in enumerate(seps)

        t_posteriors = zeros(t_max, n_reps)
        k_posteriors = zeros(t_max, n_reps)

        for rep in 1:n_reps
            mean2 = [sep, 0]
            d2 = MvNormal(mean2, C)
            data = [
                (rand() < p1 ? rand(d1, 1) : rand(d2, 1))[:]::Array{Float64,1}
                for j in 1:n
            ]

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
            "./comp_outputs/t_posteriors-mvn-dpm-sep=$sep-n=$n.jld",
            "t_posteriors",
            t_posteriors
        )

        save(
            "./comp_outputs/k_posteriors-mvn-mfm-sep=$sep-n=$n.jld",
            "k_posteriors",
            k_posteriors
        )

        println("Saved t_posteriors for sep=$sep")
        println()
    end
end
