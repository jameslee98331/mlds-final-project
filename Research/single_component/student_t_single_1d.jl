using Distributed
addprocs(5)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [7500, 10000]::Array{Int}
n_sets = 50
sets = [10, 20, 21, 44]
dofs = [1000]::Array{Int}

@everywhere function run_simulation(x, mcmc_its, mcmc_burn, t_max)
    mfm_options = BayesianMixtures.options(
        "Normal",
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

    for dof in dofs

        all_data = h5read(
            "./data_inputs/student_t/1d/single_t_1d-dof=$dof-set-$set.jld",
            "data"
        )

        for n in ns

            # create dataset with 1 single component
            data = [all_data[j]::Float64 for j in 1:n]

            # run MFM sampler
            mcmc_its = 10^5
            mcmc_burn = Int(mcmc_its / 10)
            t_max = 250
            result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

            k_posterior = BayesianMixtures.k_posterior(result)
            t_posterior = BayesianMixtures.t_posterior(result)

            # what results to store
            save(
                "./comp_outputs/student_t/1d/k_posterior-single_t_1d-dof=$dof-n=$n-set-$set.jld",
                "k_posterior",
                k_posterior
            )

            save(
                "./comp_outputs/student_t/1d/t_posterior-single_t_1d-dof=$dof-n=$n-set-$set.jld",
                "t_posterior",
                t_posterior
            )
        end
    end
end
