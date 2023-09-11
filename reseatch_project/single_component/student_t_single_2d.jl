using Distributed
addprocs(10)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD
@everywhere using Dates

ns = [1500, 2500, 5000, 7500, 10000]
n_sets = 50
sets = 1:n_sets
dofs = [2, 5, 10, 50, 1000]::Array{Int}

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
    for n in ns
        for dof in dofs
            all_data = h5read(
                "./data_inputs/student_t/2d/single_t_2d-dof=$dof-set-$set.jld",
                "data"
            )

            # create dataset with 1 single component
            data = [all_data[j, :]::Array{Float64} for j in 1:n]

            # run MFM sampler
            mcmc_its = 10^5
            mcmc_burn = 50000
            t_max = 300
            result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

            k_posterior = BayesianMixtures.k_posterior(result)
            t_posterior = BayesianMixtures.t_posterior(result)

            println(Dates.now())
            save(
                "./comp_outputs/student_t/2d/k_posterior-single_t_2d-dof=$dof-n=$n-set-$set.jld",
                "k_posterior",
                k_posterior
            )
            save(
                "./comp_outputs/student_t/2d/t_posterior-single_t_2d-dof=$dof-n=$n-set-$set.jld",
                "t_posterior",
                t_posterior
            )
        end
    end
end
