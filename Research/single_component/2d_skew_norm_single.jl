using Distributed
addprocs(12)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [2500, 100000, 1000000]
n_sets = 50

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

# iterate over ns
@sync @distributed for set in 1:n_sets
    # file_name = "./data_inputs/single_skew_normal_2d_set-$set.jld"
    # print(file_name)

    all_data = h5read(
        "./data_inputs/single_skew_normal_2d_set-$set.jld",
        "data"
    )

    for (i_n, n) in enumerate(ns)
        data = [all_data[j, :]::Array{Float64} for j in 1:n]
        result = run_simulation(data)
        k_posterior = BayesianMixtures.k_posterior(result)
        save(
            "./comp_outputs/k_posterior_single_skew_normal_2d_n=$n-set-$set.jld",
            "k_posterior",
            k_posterior
        )

        # create dataset with 1 single component:
        # 1D or 2D?

        # run MFM sampler
        # what results to store
    end
end
