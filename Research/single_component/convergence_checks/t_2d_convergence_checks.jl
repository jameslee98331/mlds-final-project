using Distributed
addprocs(10)
@everywhere using BayesianMixtures
@everywhere using HDF5
@everywhere using JLD

ns = [100, 250, 500, 750, 1000, 1500, 2500, 5000, 7500, 10000]
dof = 2
set = 8

@everywhere function run_simulation(x, mcmc_its, mcmc_burn, t_max)
    mfm_options = BayesianMixtures.options(
        "MVN",
        "MFM",
        x,
        mcmc_its,
        n_burn=mcmc_burn,
        t_max=t_max
    )
    mfm_result = BayesianMixtures.run_sampler(mfm_options)
    return mfm_result
end

all_data = h5read(
    "../data_inputs/student_t/2d/single_t_2d-dof=$dof-set-$set.jld",
    "data"
)

@sync @distributed for n in ns
    data = [all_data[j, :]::Array{Float64} for j in 1:n]

    # run MFM sampler
    mcmc_its = 10^5
    mcmc_burn = 5 * 10^4
    t_max = 100
    result = run_simulation(data, mcmc_its, mcmc_burn, t_max)
    save(
        "./t_draws-single_t_2d-dof=$dof-n=$n.jld",
        "t_draws",
        result.t
    )
end
