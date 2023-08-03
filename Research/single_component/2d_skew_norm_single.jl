using Distributed
addprocs(10)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]
n_sets = 50
sets = 1:n_sets::Array{Int}
alphas = [0, 1, 2, 5, 7]::Array{Int}

@everywhere function run_simulation(x, mcmc_its, mcmc_burn, t_max)
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
        "./data_inputs/single_skew_normal_alpha=1_2d_set-$set.jld",
        "data"
    )

    for n in ns
        for alpha in alphas

            # create dataset with 1 single component
            data = [all_data[j, :]::Array{Float64} for j in 1:n]

            # run MFM sampler
            mcmc_its = 10^5
            mcmc_burn = Int(mcmc_its / 10)
            t_max = Int(n / 2)
            result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

            k_posterior = BayesianMixtures.k_posterior(result)
            t_posterior = BayesianMixtures.t_posterior(result)

            # what results to store
            save(
                "./comp_outputs/k_posterior_single_skew_normal_2d-n=$n-alpha=$alpha-set-$set.jld",
                "k_posterior",
                k_posterior
            )

            save(
                "./comp_outputs/t_posterior_single_skew_normal_2d-n=$n-alpha=$alpha-set-$set.jld",
                "t_posterior",
                t_posterior
            )
        end
    end
end
