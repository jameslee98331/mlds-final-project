using Distributed
addprocs(10)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]
n_sets = 50
sets = 1:n_sets

@everywhere function run_simulation(x)
    mcmc_its = 10^5
    mcmc_burn = Int(mcmc_its / 10)
    t_max = 100
    mfm_options = BayesianMixtures.options(
        "MVN",
        "MFM",
        x,
        mcmc_its,
        n_burn=mcmc_burn,
        t_max=t_max
    )
    # Results of Miller and Harrison (2018)
    mfm_result = BayesianMixtures.run_sampler(mfm_options)
    return mfm_result
end

# iterate over sets
@sync @distributed for set in sets

    all_data = h5read(
        "./data_inputs/single_skew_normal_alpha=7_2d_set-$set.jld",
        "data"
    )

    for n in ns

        # create dataset with 1 single component
        data = [all_data[j, :]::Array{Float64} for j in 1:n]

        # run MFM sampler
        result = run_simulation(data)
        k_posterior = BayesianMixtures.k_posterior(result)

        # what results to store
        save(
            "./comp_outputs/k_posterior_single_skew_normal_alpha=7_2d_n=$n-set-$set.jld",
            "k_posterior",
            k_posterior
        )
    end
end
