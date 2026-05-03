##############################################################################
# topots_demo.jl
#
# A complete demonstration of every major TopoTS.jl capability, run on
# simulated data chosen to make the functionality easy to interpret.
#
# Sections
#   §1  Setup and signals
#   §2  Embedding  (ami_lag, fnn_dim, embed, embed_multivariate)
#   §3  Filtrations  (:rips, :alpha, :cech, :collapsed_rips, :cubical)
#   §4  Vectorisations  (landscape, betti_curve, persistence_image)
#   §5  Scalar statistics  (total_persistence, persistent_entropy, amplitude)
#   §6  Bootstrap confidence bands
#   §7  Hypothesis tests  (permutation_test, landscape_ttest)
#   §8  Windowed PH and CROCKER plots
#   §9  Change-point detection  (bottleneck, wasserstein, landscape scores)
#   §10 Sublevel-set PH  (sublevel_ph, windowed_sublevel_ph, periodogram_ph)
#   §11 Multivariate embedding
#   §12 Diagram kernels  (pss, pwg, sliced_wasserstein, kernel_matrix)
#   §13 Feature extraction for ML  (topo_features, TopoFeatureSpec)
#   §14 Visualisations  (all plot_* functions, saved to plots/)
##############################################################################

using Random
Random.seed!(42)

using TopoTS
using CairoMakie
using LinearAlgebra: dot, issymmetric, norm
using Statistics: mean, std, cor
using SpecialFunctions
using CechCore_jll

mkpath("plots")

##############################################################################
# §1  Synthetic signals
##############################################################################
# We build a small library of signals whose topological character is known
# a priori, so every result in the demo has a clear expected output.

N = 2_000          # samples for "long" signals
fs = 1_000.0       # sampling rate (Hz)
t  = range(0, (N - 1) / fs, length = N)

# (A) Pure sinusoid at 10 Hz  → single H₁ loop in phase space
ts_periodic   = sin.(2π .* 10 .* t) .+ 0.05 .* randn(N)

# (B) Quasiperiodic: 10 Hz + √2 × 7 Hz  → two-torus, two H₁ generators
ts_quasi      = sin.(2π .* 10 .* t) .+ sin.(2π .* sqrt(2) .* 7 .* t) .+
                0.05 .* randn(N)

# (C) Lorenz x-coordinate  → butterfly attractor, rich H₁
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end
# Integrate with simple RK4
function rk4_lorenz(n_steps, dt, p = (10.0, 28.0, 8/3))
    u = [1.0, 0.0, 0.0]
    out = zeros(n_steps, 3)
    du = zeros(3)
    for i in 1:n_steps
        k1 = similar(u); lorenz!(k1, u, p, 0); k1 .*= dt
        k2 = similar(u); lorenz!(k2, u .+ k1./2, p, 0); k2 .*= dt
        k3 = similar(u); lorenz!(k3, u .+ k2./2, p, 0); k3 .*= dt
        k4 = similar(u); lorenz!(k4, u .+ k3,    p, 0); k4 .*= dt
        u .+= (k1 .+ 2k2 .+ 2k3 .+ k4) ./ 6
        out[i, :] .= u
    end
    out
end
lorenz_traj = rk4_lorenz(N + 1_000, 0.005)
ts_lorenz   = lorenz_traj[1_001:end, 1]   # discard transient, x-channel

# (D) AR(1) noise  → no persistent H₁
ts_noise = let x = zeros(N); x[1] = randn()
    for i in 2:N; x[i] = 0.90 * x[i-1] + randn(); end; x
end

# (E) Regime-switching: periodic → noise at sample 1000  (for change-point demo)
ts_switch = vcat(
    sin.(2π .* 10 .* t[1:1_000]) .+ 0.05 .* randn(1_000),
    0.90 .* cumsum(randn(1_000)) .* 0.1   # diffuse random walk
)

println("§1  Signals constructed (N = $N samples each).")

##############################################################################
# §2  Embedding
##############################################################################

println("\n§2  Embedding")

# --- 2a. Automated lag and dimension selection ---
τ_p = optimal_lag(ts_periodic)         # AMI first local minimum
d_p = optimal_dim(ts_periodic; lag = τ_p)  # FNN criterion
println("  Periodic: optimal lag τ = $τ_p,  dim d = $d_p")

τ_q = optimal_lag(ts_quasi)
d_q = optimal_dim(ts_quasi; lag = τ_q)
println("  Quasi:    optimal lag τ = $τ_q,  dim d = $d_q")

τ_l = optimal_lag(ts_lorenz)
d_l = optimal_dim(ts_lorenz; lag = τ_l)
println("  Lorenz:   optimal lag τ = $τ_l,  dim d = $d_l")

# --- 2b. Construct TakensEmbedding structs ---
emb_p = embed(ts_periodic; lag = τ_p, dim = d_p)
emb_q = embed(ts_quasi;    lag = τ_q, dim = d_q)
emb_l = embed(ts_lorenz;   lag = τ_l, dim = d_l)

println("  emb_p: $(size(emb_p.points, 1)) points in ℝ^$d_p")
println("  emb_q: $(size(emb_q.points, 1)) points in ℝ^$d_q")
println("  emb_l: $(size(emb_l.points, 1)) points in ℝ^$d_l")

# --- 2c. Manual lag / dim for downstream reproducibility ---
# Use fixed values so later sections don't shift if parameter heuristics change
τ, d = 10, 3
emb_fixed_p = embed(ts_periodic; lag = τ, dim = d)
emb_fixed_n = embed(ts_noise;    lag = τ, dim = d)

# --- 2d. Multivariate embedding (§11 preview, full demo in §11) ---
X_mv = hcat(ts_periodic, ts_quasi)       # 2-channel: N × 2 matrix
emb_mv = embed_multivariate(X_mv; dim = 2, lag = τ)
println("  Multivariate emb: $(size(emb_mv.points, 1)) points in ℝ^$(size(emb_mv.points,2)) " *
        "($(emb_mv.n_channels) channels × dim=$(emb_mv.dim))")

##############################################################################
# §3  Filtrations
##############################################################################

println("\n§3  Filtrations")

# Subsample the periodic cloud to 400 pts for speed across all filtrations
pts_p = let e = emb_fixed_p.points
    idx = round.(Int, range(1, size(e, 1), length = 400))
    e[idx, :]
end

# --- 3a. Vietoris–Rips (default) ---
dgms_rips   = persistent_homology(pts_p; dim_max = 1, filtration = :rips)
println("  Rips   H0=$(length(dgms_rips[1])) H1=$(length(dgms_rips[2]))")

# --- 3b. Alpha complex ---
dgms_alpha  = persistent_homology(pts_p; dim_max = 1, filtration = :alpha)
println("  Alpha  H0=$(length(dgms_alpha[1])) H1=$(length(dgms_alpha[2]))")

# --- 3c. Edge-collapsed Rips ---
dgms_ecr    = persistent_homology(pts_p; dim_max = 1, filtration = :edge_collapsed)
println("  ECRips H0=$(length(dgms_ecr[1])) H1=$(length(dgms_ecr[2]))")

# --- 3d. Čech (if CechCore_jll available) ---
pts_small = pts_p[1:80, 1:2]   # Čech is exact: keep it small and low-dim
dgms_cech = persistent_homology(pts_small; dim_max = 1, filtration = :cech)
println("  Čech   H0=$(length(dgms_cech[1])) H1=$(length(dgms_cech[2]))")

# --- 3e. Cubical filtration ---
# For 1-D signals, cubical PH is exposed via sublevel_ph (see §10 for full demo).
# persistent_homology(:cubical) expects a 2-D matrix (image/grid); here we
# demonstrate it on a small discretised surface.
grid_2d = [sin(x) * cos(y)
           for x in range(0, 2π, length = 30),
               y in range(0, 2π, length = 30)]   # 30×30 Matrix{Float64}
dgms_cub = persistent_homology(grid_2d; filtration = :cubical)
println("  Cubical on 30×30 grid: H0=$(length(dgms_cub[1])) H1=$(length(dgms_cub[2]))")

# The "canonical" diagrams used in vectorisation sections below
dgms_p = persistent_homology(pts_p;         dim_max = 1, filtration = :rips)
dgms_n = persistent_homology(             # noise — few persistent features
    let e = emb_fixed_n.points
        idx = round.(Int, range(1, size(e,1), length = 400))
        e[idx, :]
    end; dim_max = 1, filtration = :rips)

##############################################################################
# §4  Vectorisations
##############################################################################

println("\n§4  Vectorisations")

# ---- shared grid: compute once from the periodic diagram and reuse ----
lam_p_ref = landscape(dgms_p, 1; n_layers = 3, n_grid = 200)
tgrid     = lam_p_ref.tgrid   # share across all landscapes below

# --- 4a. Persistence landscapes ---
lam_p = landscape(dgms_p, 1; n_layers = 3, tgrid = tgrid)
lam_n = landscape(dgms_n, 1; n_layers = 3, tgrid = tgrid)
println("  Landscape periodic: layer 1 max = $(round(maximum(lam_p.layers[1,:]), digits=4))")
println("  Landscape noise:    layer 1 max = $(round(maximum(lam_n.layers[1,:]), digits=4))")

# Arithmetic on landscapes
lam_diff = lam_p - lam_n            # pointwise difference
lam_mix  = 0.5 * lam_p + 0.5 * lam_n

# Landscape norms
println("  ‖λ_periodic‖_L2 = $(round(landscape_norm(lam_p, 2), digits=4))")
println("  ‖λ_noise‖_L2    = $(round(landscape_norm(lam_n, 2), digits=4))")

# --- 4b. Betti curves ---
bc_p = betti_curve(dgms_p, 1; n_grid = 200)
bc_n = betti_curve(dgms_n, 1; n_grid = 200)
println("  Betti curve periodic: peak β₁ = $(Int(round(maximum(bc_p.values))))")
println("  Betti curve noise:    peak β₁ = $(Int(round(maximum(bc_n.values))))")

# --- 4c. Persistence images ---
img_p = persistence_image(dgms_p, 1; sigma = 0.05, n_pixels = 20)
img_n = persistence_image(dgms_n, 1; sigma = 0.05, n_pixels = 20)
println("  Persistence image size: $(size(img_p.pixels))")

# --- 4d. Mean landscape over an ensemble of noisy periodic series ---
# (needed for bootstrap / hypothesis tests below)
ensemble_p = [sin.(2π .* 10 .* t) .+ 0.05 .* randn(N) for _ in 1:30]
ensemble_n = [let x = zeros(N); x[1] = randn()
    for i in 2:N; x[i] = 0.90*x[i-1] + randn(); end; x end for _ in 1:30]

function make_landscape(ts; tg = tgrid)
    pts = let e = embed(ts; lag = τ, dim = d).points
        idx = round.(Int, range(1, size(e,1), length = 300))
        e[idx, :]
    end
    dgm = persistent_homology(pts; dim_max = 1, filtration = :rips)
    landscape(dgm, 1; n_layers = 3, tgrid = tg)
end

lams_p = make_landscape.(ensemble_p)
lams_n = make_landscape.(ensemble_n)
lam_mean_p = mean_landscape(lams_p)
lam_mean_n = mean_landscape(lams_n)
println("  Mean landscape (30 periodic replicates) computed.")

##############################################################################
# §5  Scalar topological statistics
##############################################################################

println("\n§5  Scalar statistics")

tp_p  = total_persistence(dgms_p, 1; p = 1)
tp_n  = total_persistence(dgms_n, 1; p = 1)
println("  Total persistence  periodic=$(round(tp_p, digits=3))  noise=$(round(tp_n, digits=3))")

ent_p = persistent_entropy(dgms_p, 1)
ent_n = persistent_entropy(dgms_n, 1)
println("  Persistent entropy periodic=$(round(ent_p, digits=3))  noise=$(round(ent_n, digits=3))")

amp_p = amplitude(dgms_p, 1; p = Inf)
amp_n = amplitude(dgms_n, 1; p = Inf)
println("  Amplitude (p=∞)    periodic=$(round(amp_p, digits=3))  noise=$(round(amp_n, digits=3))")

amp2_p = amplitude(dgms_p, 1; p = 2)
amp2_n = amplitude(dgms_n, 1; p = 2)
println("  Amplitude (p=2)    periodic=$(round(amp2_p, digits=3))  noise=$(round(amp2_n, digits=3))")

##############################################################################
# §6  Bootstrap confidence bands
##############################################################################

println("\n§6  Bootstrap confidence bands")

Random.seed!(42)
band_p = confidence_band(lams_p; n_boot = 500, alpha = 0.05)
band_n = confidence_band(lams_n; n_boot = 500, alpha = 0.05)

println("  Band width at grid midpoint:")
mid = length(tgrid) ÷ 2
println("    periodic: lower=$(round(band_p.lower.layers[1,mid], digits=4))  " *
        "upper=$(round(band_p.upper.layers[1,mid], digits=4))")
println("    noise:    lower=$(round(band_n.lower.layers[1,mid], digits=4))  " *
        "upper=$(round(band_n.upper.layers[1,mid], digits=4))")

##############################################################################
# §7  Hypothesis tests
##############################################################################

println("\n§7  Hypothesis tests")

# --- 7a. Permutation test: do periodic and noise have different H₁ topology? ---
Random.seed!(42)
perm = permutation_test(lams_p, lams_n; n_perm = 999)
println("  Permutation test: observed L² dist = $(round(perm.statistic, digits=4)), " *
        "p = $(round(perm.pvalue, digits=3))")

# --- 7b. Pointwise landscape t-test: where on the scale axis do they differ? ---
tt = landscape_ttest(lams_p, lams_n; layer = 1)
n_sig = sum(tt.pvalues .< 0.05)
println("  Pointwise t-test: $n_sig / $(length(tt.pvalues)) grid points significant (α=0.05, uncorrected)")
sig_range = extrema(tt.tgrid[tt.pvalues .< 0.05])
println("  Significant scale range: $(round.(sig_range, digits=4))")

##############################################################################
# §8  Windowed PH and CROCKER plots
##############################################################################

println("\n§8  Windowed PH and CROCKER")

# Use the regime-switching signal for the most interesting CROCKER pattern
wd = windowed_ph(ts_switch;
    window     = 200,
    step       = 20,
    dim        = d,
    lag        = τ,
    dim_max    = 1,
    filtration = :rips)

println("  Windowed PH: $(length(wd)) windows on the switching signal")

cp_switch = crocker(wd; dim = 1, n_scale = 80)
println("  CROCKER surface size: $(size(cp_switch.surface))")

# Also compute on the stationary periodic signal for comparison
wd_p = windowed_ph(ts_periodic;
    window = 200, step = 20, dim = d, lag = τ, dim_max = 1)
cp_p = crocker(wd_p; dim = 1, n_scale = 80)

##############################################################################
# §9  Change-point detection
##############################################################################

println("\n§9  Change-point detection")

# Three scores on the switching signal
sc_bottle = bottleneck_score(wd,    1)
sc_wass   = wasserstein_score(wd,   1)
sc_land   = landscape_score(wd,     1; n_grid = 80, n_layers = 2)

println("  Score lengths: bottle=$(length(sc_bottle.scores))  wass=$(length(sc_wass.scores))  land=$(length(sc_land.scores))")

# Detect events — pass the ChangePointResult directly
evs_bottle = detect_changepoints(sc_bottle; threshold = :mad, n_mad = 2.5)
evs_wass   = detect_changepoints(sc_wass;   threshold = :mad, n_mad = 2.5)
evs_land   = detect_changepoints(sc_land;   threshold = :mad, n_mad = 2.5)

println("  Events detected — bottleneck: $(length(evs_bottle)), " *
        "wasserstein: $(length(evs_wass)), landscape: $(length(evs_land))")

# Print sample-level estimates
for (name, evs) in [("bottleneck", evs_bottle), ("wasserstein", evs_wass), ("landscape", evs_land)]
    locs = [round(Int, ev.time) for ev in evs]
    println("    $name events at samples: $locs  (true change-point: 1000)")
end

##############################################################################
# §10 Sublevel-set PH
##############################################################################

println("\n§10 Sublevel-set PH")

# --- 10a. Direct sublevel PH on a short signal ---
f_osc = sin.(range(0, 8π, length = 600)) .+ 0.1 .* randn(600)
dgm_sub = sublevel_ph(f_osc)
println("  sublevel_ph: H0=$(length(dgm_sub.H0)) finite pairs, H1=$(length(dgm_sub.H1))")

# With extended persistence
dgm_ext = sublevel_ph(f_osc; extended = true)
println("  Extended:    H0=$(length(dgm_ext.H0)) H1=$(length(dgm_ext.H1))")

# --- 10b. Periodogram PH (spectral TDA) ---
sig_spectral = sin.(2π .* 50 .* t[1:1000]) .+ 0.2 .* randn(1000)
dgm_spec = periodogram_ph(sig_spectral; bw = 5)
println("  periodogram_ph: $(length(dgm_spec.H0)) spectral H0 pairs")

# --- 10c. Windowed sublevel PH → fast change-point detection ---
wd_sub = windowed_sublevel_ph(ts_switch; window = 200, step = 20)
sc_sub = landscape_score(wd_sub, 0)
evs_sub = detect_changepoints(sc_sub; threshold = :mad, n_mad = 2.5)
println("  Windowed sublevel events: $(length(evs_sub)) " *
        "at samples $(round.(Int, [ev.time for ev in evs_sub]))")

##############################################################################
# §11 Multivariate embedding
##############################################################################

println("\n§11 Multivariate embedding")

# Build a 2-channel signal: channel 1 periodic, channel 2 quasiperiodic
X_mv2 = hcat(ts_periodic[1:1_000], ts_quasi[1:1_000])   # 1000 × 2
emb_mv2 = embed_multivariate(X_mv2; dim = 2, lag = 8)
println("  Multivariate embedding: $(size(emb_mv2.points, 1)) points in " *
        "ℝ^$(size(emb_mv2.points, 2)) ($(emb_mv2.n_channels) channels × dim=2)")

# PH on the joint point cloud (subsample for speed)
pts_mv = let e = emb_mv2.points
    idx = round.(Int, range(1, size(e, 1), length = 300))
    e[idx, :]
end
dgms_mv = persistent_homology(pts_mv; dim_max = 1, filtration = :rips)
println("  Joint H1 diagram: $(length(dgms_mv[2])) points " *
        "(cf. single-channel $(length(dgms_p[2])))")

##############################################################################
# §12 Diagram kernels
##############################################################################

println("\n§12 Diagram kernels")

# Build a small collection: 10 periodic + 10 noise diagrams
function quick_dgm(ts)
    pts = let e = embed(ts; lag = τ, dim = d).points
        idx = round.(Int, range(1, size(e,1), length = 200))
        e[idx, :]
    end
    persistent_homology(pts; dim_max = 1, filtration = :rips)
end

coll_p = [quick_dgm(sin.(2π .* 10 .* t) .+ 0.05 .* randn(N)) for _ in 1:10]
coll_n = [quick_dgm(let x = zeros(N); x[1] = randn()
    for i in 2:N; x[i] = 0.90*x[i-1] + randn(); end; x end) for _ in 1:10]
all_dgms = vcat(coll_p, coll_n)

# --- 12a. Pointwise kernel values ---
k_pss  = pss_kernel(coll_p[1][2], coll_n[1][2]; sigma = 0.3)
k_pwg  = pwg_kernel(coll_p[1][2], coll_n[1][2]; sigma = 0.3, C = 2.0)
k_sw   = sliced_wasserstein_kernel(coll_p[1][2], coll_n[1][2]; sigma = 0.5, n_directions = 100)
println("  K_pss(periodic, noise)  = $(round(k_pss, digits=5))")
println("  K_pwg(periodic, noise)  = $(round(k_pwg, digits=5))")
println("  K_sw( periodic, noise)  = $(round(k_sw,  digits=5))")

# Self-similarity (should be larger)
k_pss_self = pss_kernel(coll_p[1][2], coll_p[2][2]; sigma = 0.3)
println("  K_pss(periodic, periodic) = $(round(k_pss_self, digits=5))  (expect > cross)")

# --- 12b. Gram matrices ---
Random.seed!(42)
K_pss = kernel_matrix(all_dgms, 1; kernel = :pss, sigma = 0.3)
K_sw  = kernel_matrix(all_dgms, 1; kernel = :sliced_wasserstein, sigma = 0.5)
println("  Gram matrix K_pss size: $(size(K_pss))  symmetric=$(issymmetric(round.(K_pss, digits=8)))")

# Centroid classifier accuracy (leave-all-in for illustration)
y = vcat(fill(1, 10), fill(-1, 10))
c1 = mean(K_pss[1:10,  :], dims = 1)[1, :]
c2 = mean(K_pss[11:20, :], dims = 1)[1, :]
preds = [dot(K_pss[i, :], c1) >= dot(K_pss[i, :], c2) ? 1 : -1 for i in 1:20]
acc = mean(preds .== y)
println("  Centroid classifier (K_pss, H₁): accuracy = $(round(100acc, digits=1))%")

# --- 12c. Exact Wasserstein distance ---
to_tuples(dgm) = [(Float64(p[1]), Float64(p[2])) for p in dgm if isfinite(p[2])]
d_w12 = wasserstein_distance(to_tuples(coll_p[1][2]), to_tuples(coll_n[1][2]); p = 2)
println("  W₂(periodic_1, noise_1) = $(round(d_w12, digits=4))")

##############################################################################
# §13 Feature extraction for ML
##############################################################################

println("\n§13 Feature extraction")

# --- 13a. Basic spec ---
spec = TopoFeatureSpec(
    dim_max            = 1,
    dim                = d,
    lag                = τ,
    filtration         = :rips,
    use_landscape      = true,
    n_landscape_layers = 3,
    n_landscape_grid   = 50,
    use_betti          = true,
    n_betti_grid       = 50,
    use_stats          = true,
    use_image          = false,
)

feat_p = topo_features(ts_periodic[1:500]; spec = spec)
feat_n = topo_features(ts_noise[1:500];    spec = spec)
println("  Feature vector length: $(length(feat_p))")
println("  Feature names (first 8): $(feature_names(spec)[1:8])")
println("  ‖feat_periodic - feat_noise‖₂ = $(round(norm(feat_p .- feat_n), digits=3))")

# --- 13b. From a pre-computed DiagramCollection (avoids recomputing PH) ---
feat_from_dgm = topo_features(dgms_p; spec = spec)
println("  topo_features from DiagramCollection: length $(length(feat_from_dgm))")

# --- 13c. Feature matrix for a small classification problem ---
n_per_class = 20
gbm(n) = cumsum(0.001 .+ 0.02 .* randn(n))
ou(n)  = (x = zeros(n); for i in 2:n
              x[i] = x[i-1] + 0.1*(0.0 - x[i-1]) + 0.02*randn()
          end; x)

spec_ml = TopoFeatureSpec(
    dim_max = 1, dim = 2, lag = 5, use_landscape = true,
    n_landscape_layers = 2, n_landscape_grid = 30, use_stats = true,
    use_betti = false, use_image = false,
)

X_gbm = reduce(vcat, topo_features(gbm(300); spec = spec_ml)' for _ in 1:n_per_class)
X_ou  = reduce(vcat, topo_features( ou(300); spec = spec_ml)' for _ in 1:n_per_class)
X_ml  = vcat(X_gbm, X_ou)
y_ml  = vcat(fill(1, n_per_class), fill(-1, n_per_class))

# Nearest-centroid classifier
c_gbm = mean(X_gbm, dims = 1)[1, :]
c_ou  = mean(X_ou,  dims = 1)[1, :]
preds_ml = [norm(X_ml[i, :] .- c_gbm) <= norm(X_ml[i, :] .- c_ou) ? 1 : -1
            for i in 1:2n_per_class]
acc_ml = mean(preds_ml .== y_ml)
println("  Nearest-centroid on GBM vs OU ($n_per_class per class): accuracy = $(round(100acc_ml, digits=1))%")

##############################################################################
# §14 Visualisations
##############################################################################

println("\n§14 Visualisations  (writing to plots/)")

# --- 14a. Persistence diagram ---
fig_diag = plot_diagram(dgms_p, 1; title = "H₁ Persistence Diagram — periodic signal")
save("plots/diag_periodic.pdf", fig_diag)

# Multi-diagram panel (four signals side by side)
dgms_q  = persistent_homology(
    let e = embed(ts_quasi;  lag = τ, dim = d).points
        idx = round.(Int, range(1, size(e,1), length = 400)); e[idx, :]
    end; dim_max = 1, filtration = :rips)
dgms_lo = persistent_homology(
    let e = embed(ts_lorenz; lag = τ, dim = d).points
        idx = round.(Int, range(1, size(e,1), length = 400)); e[idx, :]
    end; dim_max = 1, filtration = :rips)

fig_multi = let
    labels = ["Periodic", "Quasiperiodic", "Lorenz", "AR(1) noise"]
    colls  = [dgms_p, dgms_q, dgms_lo, dgms_n]
    f = Figure(size = (1200, 320))
    for (i, (dgm, lbl)) in enumerate(zip(colls, labels))
        ax = Axis(f[1, i], title = "H\u2081 \u2014 $lbl",
                  xlabel = "birth", ylabel = "death", aspect = 1)
        for p in dgm[2]
            isfinite(p[2]) || continue
            scatter!(ax, [p[1]], [p[2]]; color = :steelblue,
                     markersize = 6, strokewidth = 0)
        end
        b_vals = [p[1] for p in dgm[2] if isfinite(p[2])]
        d_vals = [p[2] for p in dgm[2] if isfinite(p[2])]
        if !isempty(b_vals)
            lo = min(minimum(b_vals), minimum(d_vals))
            hi = max(maximum(b_vals), maximum(d_vals))
            lines!(ax, [lo, hi], [lo, hi]; color = :gray60, linestyle = :dash)
        end
    end
    f
end
save("plots/diag_multi.pdf", fig_multi)

# --- 14b. Barcode ---
fig_bar = plot_barcode(dgms_p, 1; title = "H₁ Barcode — periodic signal")
save("plots/barcode_periodic.pdf", fig_bar)

# --- 14c. Persistence landscape ---
fig_lam = plot_landscape(lam_p; title = "H₁ Persistence Landscape — periodic (3 layers)")
save("plots/landscape_periodic.pdf", fig_lam)

# Bootstrap bands plot: mean ± 95% band for both groups
fig_band = let
    f = Figure(size = (900, 400))
    ax1 = Axis(f[1, 1], title = "Bootstrap band — periodic",
               xlabel = "filtration ε", ylabel = "λ₁(ε)")
    ax2 = Axis(f[1, 2], title = "Bootstrap band — AR(1) noise",
               xlabel = "filtration ε")
    for (ax, band, col) in [(ax1, band_p, :steelblue), (ax2, band_n, :tomato)]
        band!(ax, tgrid, band.lower.layers[1,:], band.upper.layers[1,:];
              color = (col, 0.25))
        lines!(ax, tgrid, band.mean.layers[1,:]; color = col, linewidth = 2)
    end
    f
end
save("plots/bootstrap_bands.pdf", fig_band)

# --- 14d. Betti curve ---
fig_bc = plot_betti_curve(bc_p; title = "H₁ Betti Curve — periodic signal")
save("plots/betti_periodic.pdf", fig_bc)

# Overlaid Betti curves for all four signals
fig_bc4 = let
    bc_q  = betti_curve(dgms_q,  1; n_grid = 200)
    bc_lo = betti_curve(dgms_lo, 1; n_grid = 200)
    f = Figure(size = (700, 400))
    ax = Axis(f[1, 1], title = "H₁ Betti Curves",
              xlabel = "filtration ε", ylabel = "β₁(ε)")
    for (bc, label, col) in [
            (bc_p,  "Periodic",      :steelblue),
            (bc_q,  "Quasiperiodic", :darkorange),
            (bc_lo, "Lorenz",        :forestgreen),
            (bc_n,  "AR(1) noise",   :tomato)]
        lines!(ax, bc.tgrid, bc.values; label = label, color = col, linewidth = 2)
    end
    axislegend(ax; position = :rt)
    f
end
save("plots/betti_four_signals.pdf", fig_bc4)

# --- 14e. Pointwise t-test result ---
fig_ttest = let
    f = Figure(size = (700, 350))
    ax1 = Axis(f[1, 1], title = "Mean Landscape Layer 1 — periodic vs noise",
               xlabel = "filtration ε", ylabel = "λ₁(ε)")
    lines!(ax1, tgrid, lam_mean_p.layers[1,:]; color = :steelblue, label = "periodic", linewidth = 2)
    lines!(ax1, tgrid, lam_mean_n.layers[1,:]; color = :tomato, label = "AR(1)", linewidth = 2)
    axislegend(ax1)
    ax2 = Axis(f[2, 1], title = "Pointwise t-test −log₁₀(p)",
               xlabel = "filtration ε", ylabel = "−log₁₀(p)")
    logp = -log10.(max.(tt.pvalues, 1e-10))
    lines!(ax2, tt.tgrid, logp; color = :purple, linewidth = 2)
    hlines!(ax2, [-log10(0.05)]; color = :black, linestyle = :dash,
            label = "α = 0.05")
    axislegend(ax2)
    f
end
save("plots/ttest_landscapes.pdf", fig_ttest)

# --- 14f. CROCKER plots ---
fig_ck_switch = plot_crocker(cp_switch;
    title = "CROCKER — regime-switching signal (periodic → random walk)")
save("plots/crocker_switch.pdf", fig_ck_switch)

fig_ck_p = plot_crocker(cp_p; title = "CROCKER — stationary periodic")
save("plots/crocker_periodic.pdf", fig_ck_p)

# --- 14g. Change-point score ---
fig_cp = plot_changepoint_score(sc_land; events = evs_land,
    title = "Landscape Change-Point Score — switching signal (true CP: sample 1000)")
save("plots/changepoint_score.pdf", fig_cp)

# All three scores in one figure
fig_cp3 = let
    f = Figure(size = (900, 600))
    titles = ["Bottleneck score", "Wasserstein score", "Landscape score"]
    scores = [sc_bottle, sc_wass, sc_land]
    events = [evs_bottle, evs_wass, evs_land]
    for (i, (sc, ev, ttl)) in enumerate(zip(scores, events, titles))
        ax = Axis(f[i, 1], title = ttl, ylabel = "score",
                  xlabel = i == 3 ? "window index" : "")
        lines!(ax, sc.scores; color = :steelblue, linewidth = 1.5)
        for e in ev
            vlines!(ax, [e.index]; color = :tomato, linestyle = :dash)
        end
        vlines!(ax, [50]; color = :black, linestyle = :dot,
                label = "true CP")   # window ~50 corresponds to sample ~1000
    end
    f
end
save("plots/changepoint_three_scores.pdf", fig_cp3)

# --- 14h. Gram matrix heatmap ---
fig_gram = let
    f = Figure(size = (500, 460))
    ax = Axis(f[1, 1], title = "PSS Kernel Gram Matrix (H₁)",
              xlabel = "series index", ylabel = "series index")
    hm = heatmap!(ax, K_pss; colormap = :viridis)
    Colorbar(f[1, 2], hm)
    # Mark the class boundary
    vlines!(ax, [10.5]; color = :white, linewidth = 2, linestyle = :dash)
    hlines!(ax, [10.5]; color = :white, linewidth = 2, linestyle = :dash)
    f
end
save("plots/gram_pss.pdf", fig_gram)

println("  Saved: diag_periodic, diag_multi, barcode_periodic,")
println("         landscape_periodic, bootstrap_bands, betti_periodic,")
println("         betti_four_signals, ttest_landscapes, crocker_switch,")
println("         crocker_periodic, changepoint_score, changepoint_three_scores,")
println("         gram_pss")

##############################################################################
# Summary
##############################################################################

println("""
\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TopoTS.jl demo complete.  Modules exercised:

  Embedding          ami_lag, fnn_dim, optimal_lag, optimal_dim,
                     embed, embed_multivariate
  Filtration         :rips, :alpha, :collapsed_rips, :cech, :cubical
  BettiCurves        betti_curve
  Landscapes         landscape, mean_landscape, landscape_norm
  PersistenceImages  persistence_image
  TopoStats          total_persistence, persistent_entropy, amplitude
  Bootstrap          confidence_band
  HypothesisTests    permutation_test, landscape_ttest
  Windowed           windowed_ph, WindowedDiagrams
  CROCKER            crocker, CROCKERPlot
  ChangePoint        bottleneck_score, wasserstein_score,
                     landscape_score, detect_changepoints
  Sublevel           sublevel_ph (standard + extended),
                     windowed_sublevel_ph, periodogram_ph
  Multivariate       embed_multivariate
  DiagramKernels     pss_kernel, pwg_kernel,
                     sliced_wasserstein_kernel, kernel_matrix,
                     wasserstein_distance
  Features           topo_features, TopoFeatureSpec, feature_names
  Visualisations     plot_diagram, plot_diagram_multi, plot_barcode,
                     plot_landscape, plot_betti_curve, plot_crocker,
                     plot_changepoint_score
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
