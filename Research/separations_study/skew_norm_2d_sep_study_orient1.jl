using Distributed
addprocs(10)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD
@everywhere using Dates

ns = [250, 500, 750, 1000, 1500] * 2
n_sets = 50
sets = 1:n_sets
seps = 1:5
alpha = 7

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

@sync @distributed for set in sets
    for sep in seps
        all_data = h5read(
            "./data_inputs/skew_norm/2d/skew_normal_2d-alpha=$alpha-sep=$sep-orient=1-set-$set.jld",
            "data"
        )
        for n in ns
            # create dataset with 1 single component
            data = [all_data[j, :]::Array{Float64} for j in 1:n]

            # run MFM sampler
            mcmc_its = 10^5
            mcmc_burn = 50000
            t_max = 250
            result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

            k_posterior = BayesianMixtures.k_posterior(result)
            t_posterior = BayesianMixtures.t_posterior(result)

            # what results to store
            println(Dates.now())
            save(
                "./comp_outputs/skew_norm/2d/k_posterior-skew_normal_2d-alpha=$alpha-sep=$sep-n=$n-orient=1-set-$set.jld",
                "k_posterior",
                k_posterior
            )
            save(
                "./comp_outputs/skew_norm/2d/t_posterior-skew_normal_2d-alpha=$alpha-sep=$sep-n=$n-orient=1-set-$set.jld",
                "t_posterior",
                t_posterior
            )
        end
    end
end
