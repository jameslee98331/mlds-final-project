# Supporting parameters and functions for skew-normal simulation example

# Stuff for mixture weights
# maximum number of components
m = 10

# Gamma parameters
using Distributions

a = 1 / m

# prior on weights (before conditioning on s>0)
G = Gamma(a, 1 / 1)

# Poisson parameter of the limiting prior distribution on k as m -> Inf
lambda = 1.0

# cutoff (Note: This choice makes p(k) \propto Binomial(k|m,lambda/m)I(k>0).)
c = quantile(G, 1 - lambda / m)

# scale for weight proposals
sigma = 0.25

# prior on latent weights v_i
log_v_prior(v_i) = logpdf(G, v_i)


# Stuff for mixture components
normpdf(x, m, l) = sqrt(l / (2 * pi)) * exp(-0.5 * l * (x - m) .* (x - m))
normlogpdf(x, m, s) = -0.5 * log(2 * pi) - log(s) - 0.5 * (x - m) * (x - m) / (s * s)
likelihood(x, t) = normpdf(x, t[1], t[2])  # t = [mean, precision]

mvnormpdf(x, mu, Sigma) = pdf(MvNormal(mu, Sigma), x)
likelihood(x, mu, Sigma) = mvnormpdf(x, mu, Sigma)



# prior (base measure) on component params, [mean, log(precision)]
m0m = 0
s0m = 5
m0l = 0
s0l = 2
log_theta_prior(t) = normlogpdf(t[1], m0m, s0m) + normlogpdf(log(t[2]), m0l, s0l)

# proposal distribution for component param moves
sm = 0.2 * s0m
sl = 0.2
theta_prop(t) = Float64[t[1]+sm*randn(), t[2]*exp(randn() * sl)]
log_theta_prop(t, tp) = normlogpdf(tp[1], t[1], sm) + normlogpdf(log(tp[2]), log(t[2]), sl)

mu = [0, 0]

# initialize params for m new components
new_thetas(m) = [Float64[m0m, exp(m0l)] for i = 1:m]

# Sampler code
include("core_2d.jl")
