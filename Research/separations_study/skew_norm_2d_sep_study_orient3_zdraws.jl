using Distributed
addprocs(5)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD
@everywhere using Dates

set = 8
alpha = 7
seps = 1:5
ns = [100, 250, 500, 750, 1000] * 2

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

@sync @distributed for n in ns
    for sep in seps
        all_data = h5read(
            "./data_inputs/skew_norm/2d/skew_normal_2d-alpha=$alpha-sep=$sep-orient=3-set-$set.jld",
            "data"
        )

        # create dataset with 1 single component
        data = [all_data[j, :]::Array{Float64} for j in 1:n]

        # run MFM sampler
        mcmc_its = 10^5
        mcmc_burn = Int(mcmc_its / 10)
        t_max = 100
        result = run_simulation(data, mcmc_its, mcmc_burn, t_max)

        # what results to store
        println(Dates.now())
        save(
            "./comp_outputs/skew_norm/2d/z_draws-skew_normal_2d-alpha=$alpha-sep=$sep-n=$n-orient=3-set-$set.jld",
            "z",
            result.z
        )
    end
end
