using Distributed
addprocs(10)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [1000]
n_sets = 50
sets = 1:n_sets
sigmas = [1, 2, 3, 4, 5]::Array{Int}

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

# iterate over ns
@sync @distributed for set in sets

    for sigma in sigmas

        all_data = h5read(
            "./data_inputs/laplace/2d/single_laplace_2d-sigma=$sigma-set-$set.jld",
            "data"
        )

        for n in ns

            # create dataset with 1 single component
            data = [all_data[j, :]::Array{Float64} for j in 1:n]

            # run MFM sampler
            mcmc_its = 10^5
            mcmc_burn = Int(mcmc_its / 10)
            t_max = 250
            result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

            k_posterior = BayesianMixtures.k_posterior(result)
            t_posterior = BayesianMixtures.t_posterior(result)

            # what results to store
            save(
                "./comp_outputs/laplace/2d/k_posterior-single_laplace_2d-sigma=$sigma-n=$n-set-$set.jld",
                "k_posterior",
                k_posterior
            )

            save(
                "./comp_outputs/laplace/2d/t_posterior-single_laplace_2d-sigma=$sigma-n=$n-set-$set.jld",
                "t_posterior",
                t_posterior
            )
        end
    end
end
