# Scalar nonlinear state-estimation benchmark

Compare KF (deliberately misspecified linear surrogate), EKF, UKF, bootstrap particle filter (BPF), and an **illustrative** adaptive Gaussian–Cauchy resample-move particle filter (AGCR-PF).

## Model

- State: `x[k] = 0.55*x[k-1] + 18*x[k-1]/(1+x[k-1]^2) + 7*cos(0.7*k) + w[k]`
- Measurement: `y[k] = x[k]^2/15 + v[k]`
- Independent `w ~ Normal(0,1)` and `v ~ Normal(0,0.01)`.
- `x[-1] = 0`, initial filter estimate `0` with covariance `5`.

## Run

```bash
pip install numpy matplotlib
python compare_filters.py --runs 30 --steps 60 --particles 200 --seed 2026
```

Results: `results/per_run.csv`, `results/trajectories.png`, `results/rmse_boxplot.png`.

## Important interpretation

- The **KF is not the optimal Kalman filter** for this nonlinear system; it uses a fixed linear surrogate and is included as a deliberately misspecified baseline.
- EKF and UKF are Gaussian approximations. The squared measurement is sign-ambiguous and can yield multimodal posteriors.
- BPF uses 200 particles and systematic resampling when ESS drops below N/2.
- AGCR-PF is an **illustrative design**, not a confirmed implementation of the researcher's exact unpublished algorithm: it resamples every step and applies one Gaussian/Cauchy mixture random-walk Metropolis move (20% Cauchy proposals) with a kernel-density approximation to the predictive prior. It does not dynamically learn mixture weight; the name is a placeholder until the precise adaptive update is supplied.
- The likelihood and model are identical across methods; randomness and CPU timing vary by machine. Runtime includes Python implementation overhead and is not a hardware-independent complexity comparison.
- The experiment is not a general proof that one method dominates another. Multiple seeds, particle counts and models should be tested.
