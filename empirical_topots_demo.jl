##############################################################################
# empirical_topots_demo.jl
#
# Reproducibility script for the empirical illustrations in:
#   "TopoTS: Topological Data Analysis for Time Series in Julia"
#   Journal of Statistical Software
#
# Sections
#   §1  Setup and dependencies
#   §2  NASA IMS bearing dataset
#         - Distance sequences (Pipelines A and B)
#         - Change-point detection (Andrews sup-F)
#         - Figures: nasa_set1.pdf, nasa_set2.pdf,
#                    nasa_set1_m5.pdf, nasa_set2_m5.pdf
#         - CROCKER plots: crocker_set1.pdf, crocker_set2.pdf, crocker_set3.pdf
#   §3  Victorian electricity demand forecasting
#         - Data download (automatic)
#         - Topo feature computation
#         - XGBoost walk-forward evaluation (h=1, h=48)
#         - Table: results/elec_forecasting.csv
#
# DATA REQUIREMENTS
#   NASA IMS: Not bundled (~6 GB). Download from:
#     https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
#
#   IMPORTANT — directory naming quirk in the NASA archive:
#     After unpacking, experiment 3 resides in  4th_test/txt/  (not 3rd_test/).
#     Before running this script, copy those files into IMS/3rd_test/:
#       Unix/macOS:  cp -r 4th_test/txt/* IMS/3rd_test/
#       Windows:     Copy-Item 4th_test\txt\* IMS\3rd_test\ -Recurse
#
#   Expected paths after setup (can be overridden via ENV vars or ARGS):
#     IMS/1st_test/   — 2156 trial files, 8 channels
#     IMS/2nd_test/   — 984  trial files, 4 channels
#     IMS/3rd_test/   — 6324 trial files (copied from 4th_test/txt/)
#
#   vic_elec: Downloaded automatically from GitHub (tsibbledata).
#
# USAGE
#   julia --threads=auto empirical_topots_demo.jl
#   julia --threads=auto empirical_topots_demo.jl /path/to/1st_test /path/to/2nd_test /path/to/3rd_test
#
# Intermediate results are checkpointed to checkpoints/ so that
# expensive computations are not repeated on re-runs.
# All output figures are written to plots/empirical/.
##############################################################################

##############################################################################
# §1  Setup
##############################################################################

using Random
Random.seed!(42)

using TopoTS
using FFTW
using XGBoost
using DelimitedFiles, Statistics, StatsBase
using CairoMakie
using CSV, DataFrames, Downloads
using Serialization
using Printf

mkpath("plots/empirical")
mkpath("checkpoints/nasa")
mkpath("checkpoints/elec")
mkpath("results")

# ── Checkpoint helpers ────────────────────────────────────────────────────────

function cached(fn::Function, key::String, dir::String = "checkpoints/nasa")
    path = joinpath(dir, key * ".jls")
    if isfile(path)
        println("  [ckpt] loading $key ...")
        return deserialize(path)
    end
    result = fn()
    serialize(path, result)
    println("  [ckpt] saved $key ($(filesize(path) ÷ 1024) KB)")
    return result
end

##############################################################################
# §2  NASA IMS bearing dataset
##############################################################################

println("\n" * "="^70)
println("§2  NASA IMS bearing dataset")
println("="^70)

# ── 2.0  Data paths ───────────────────────────────────────────────────────────

set1_dir = get(ENV, "IMS_SET1", length(ARGS) >= 1 ? ARGS[1] : "IMS/1st_test")
set2_dir = get(ENV, "IMS_SET2", length(ARGS) >= 2 ? ARGS[2] : "IMS/2nd_test")
set3_dir = get(ENV, "IMS_SET3", length(ARGS) >= 3 ? ARGS[3] : "IMS/3rd_test")

for (d, name) in [(set1_dir, "Set 1"), (set2_dir, "Set 2")]
    isdir(d) || error("""
NASA IMS $name directory not found: $d

Download the IMS bearing dataset (~6 GB) from:
  https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

IMPORTANT — directory naming quirk in the NASA archive:
  The downloaded archive places experiment 3 data in  4th_test/txt/
  rather than 3rd_test/.  Before running this script, copy those files:
    Unix/macOS:  cp -r 4th_test/txt/* IMS/3rd_test/
    Windows:     Copy-Item 4th_test\\txt\\* IMS\\3rd_test\\ -Recurse

Expected directory structure after setup:
  IMS/1st_test/   — 2156 trial files, 8 channels
  IMS/2nd_test/   — 984  trial files, 4 channels
  IMS/3rd_test/   — 6324 trial files (copied from 4th_test/txt/)

Or supply paths directly as command-line arguments:
  julia --threads=auto empirical_topots_demo.jl /path/1st_test /path/2nd_test /path/3rd_test
""")
end
has_set3 = isdir(set3_dir)
has_set3 || @warn "Set 3 directory not found ($set3_dir) — Set 3 figures will be skipped."

# ── 2.1  Data loading ─────────────────────────────────────────────────────────

function load_channel(dir::String, col::Int)
    files = sort(filter(f -> !isdir(f) && filesize(f) > 0 &&
                             !startswith(basename(f), "."),
                        readdir(dir; join = true)))
    isempty(files) && error("No files in $dir")
    @info "$(length(files)) trials, channel $col — $(basename(dir))"
    [readdlm(f, Float64)[:, col] for f in files]
end

# ── 2.2  Spectral pipeline (Pipeline A) ───────────────────────────────────────

function smoothed_periodogram(sig; bw = 5, keep = 4)
    P = abs2.(rfft(sig)) ./ length(sig)
    P = [mean(P[max(1, i-bw):min(length(P), i+bw)]) for i in 1:length(P)]
    return P[1:keep:end]
end

# A-Sc1: cumulative average periodogram vs next trial (L1 norm)
function pipe_A_scenario1(ch::Vector{Vector{Float64}}, ck::String; bw = 5)
    R  = length(ch)
    Ps = cached(ck * "_A1_Ps") do
        result = Vector{Vector{Float64}}(undef, R)
        done   = Threads.Atomic{Int}(0)
        println("    precomputing periodograms (R=$R, $(Threads.nthreads()) threads):")
        Threads.@threads for i in 1:R
            result[i] = smoothed_periodogram(ch[i]; bw)
            n = Threads.atomic_add!(done, 1) + 1
            n % 200 == 0 && (print("\r      $n/$R"); flush(stdout))
        end
        println(); result
    end
    cached(ck * "_A1_D") do
        D  = zeros(R - 1)
        S  = copy(Ps[1])
        for r in 1:R-1
            D[r]  = sum(abs, S ./ r .- Ps[r+1])
            S    .+= Ps[r+1]
        end
        D
    end
end

# A-Sc2: consecutive periodogram topological H0 distances
function pipe_A_scenario2(ch::Vector{Vector{Float64}}, ck::String; bw = 5)
    R    = length(ch)
    dgms = cached(ck * "_A2_dgms") do
        result = Vector{Vector{Tuple{Float64,Float64}}}(undef, R)
        done   = Threads.Atomic{Int}(0)
        println("    computing spectral diagrams (R=$R):")
        Threads.@threads for i in 1:R
            result[i] = [(Float64(b), Float64(d))
                         for (b, d) in sublevel_ph(smoothed_periodogram(ch[i]; bw)).H0]
            n = Threads.atomic_add!(done, 1) + 1
            n % 200 == 0 && (print("\r      $n/$R"); flush(stdout))
        end
        println(); result
    end
    cached(ck * "_A2_D") do
        D  = zeros(R - 1)
        done = Threads.Atomic{Int}(0)
        Threads.@threads for r in 1:R-1
            D[r] = wasserstein_distance(dgms[r], dgms[r+1])
            n = Threads.atomic_add!(done, 1) + 1
            n % 200 == 0 && (print("\r      $n/$(R-1)"); flush(stdout))
        end
        println(); D
    end
end

# ── 2.3  Phase-space pipeline (Pipeline B) ───────────────────────────────────

to_pairs(dgm) = [(Float64(p[1]), Float64(p[2])) for p in dgm if isfinite(p[2])]

function pipe_B(ch::Vector{Vector{Float64}}, ck::String;
                sub::Int = 20, n_ref::Int = 10, dim::Int = 3)
    ref = reduce(vcat, [c[1:sub:end] for c in ch[1:n_ref]])
    τ   = optimal_lag(ref)
    @info "  Pipeline B: τ* = $τ ($(Threads.nthreads()) threads)"
    R    = length(ch)
    dgms = cached(ck * "_B_dgms") do
        result = Vector{Vector{Tuple{Float64,Float64}}}(undef, R)
        done   = Threads.Atomic{Int}(0)
        println("    computing phase-space PH (R=$R):")
        Threads.@threads for i in 1:R
            ds       = ch[i][1:sub:end]
            emb      = embed(ds; dim = dim, lag = τ)
            ph       = persistent_homology(emb; dim_max = 1)
            result[i] = to_pairs(ph[2])
            n = Threads.atomic_add!(done, 1) + 1
            n % 200 == 0 && (print("\r      $n/$R"); flush(stdout))
        end
        println(); result
    end
    cached(ck * "_B_D") do
        D  = zeros(R - 1)
        done = Threads.Atomic{Int}(0)
        Threads.@threads for r in 1:R-1
            D[r] = wasserstein_distance(dgms[r], dgms[r+1])
            n = Threads.atomic_add!(done, 1) + 1
            n % 200 == 0 && (print("\r      $n/$(R-1)"); flush(stdout))
        end
        println(); D
    end
end

# ── 2.4  Andrews sup-F change-point detection ─────────────────────────────────

function andrews_supF(D::Vector{Float64}; r0::Int, alpha::Float64 = 0.05)
    R    = length(D)
    S    = cumsum(D)
    S2   = cumsum(D .^ 2)
    mu_n = S[R] / R
    rss0 = S2[R] - R * mu_n^2

    best_F = -Inf
    best_r = div(R, 2)
    for r in r0:R-r0
        n1   = r;          n2   = R - r
        mu1  = S[r] / n1;  mu2  = (S[R] - S[r]) / n2
        rss  = (S2[r] - n1*mu1^2) + ((S2[R]-S2[r]) - n2*mu2^2)
        F    = ((rss0 - rss) / 2) / (rss / (R - 2))
        if F > best_F; best_F = F; best_r = r; end
    end

    # Andrews (1993) critical values, 1 restriction, 15% trimming
    cv   = get(Dict(0.10 => 7.12, 0.05 => 8.85, 0.01 => 12.16), alpha, 8.85)
    sig  = best_F > cv
    (r_star = best_r, sup_F = best_F, cv = cv, significant = sig)
end

# ── 2.5  Analyse one bearing ──────────────────────────────────────────────────

function nasa_analyse(ch; r0, label, ck)
    R = length(ch)
    println("\n  $label  (R=$R, r₀=$r0)")
    D_A1 = pipe_A_scenario1(ch, ck)
    D_A2 = pipe_A_scenario2(ch, ck)
    D_B  = pipe_B(ch, ck)
    m_A1 = andrews_supF(D_A1; r0)
    m_A2 = andrews_supF(D_A2; r0)
    m_B  = andrews_supF(D_B;  r0)
    star(s) = s ? "*" : " "
    @printf("    A-Sc1: r*=%4d/%d (%.2f)%s  A-Sc2: r*=%4d/%d (%.2f)%s  B: r*=%4d/%d (%.2f)%s\n",
            m_A1.r_star, R, m_A1.r_star/R, star(m_A1.significant),
            m_A2.r_star, R, m_A2.r_star/R, star(m_A2.significant),
            m_B.r_star,  R, m_B.r_star/R,  star(m_B.significant))
    (; D_A1, D_A2, D_B, m_A1, m_A2, m_B, r0, R, label)
end

# ── 2.6  Distance sequence figures ───────────────────────────────────────────

function bearing_panel!(ax, br; show_legend = false)
    μ_A1 = mean(br.D_A1[1:br.r0])
    μ_A2 = mean(br.D_A2[1:br.r0])
    μ_B  = mean(br.D_B[1:br.r0])
    idx  = 2:length(br.D_A1)+1
    lines!(ax, idx, br.D_A1 ./ μ_A1; color = :royalblue,  label = "A Sc.1 (cumul L1)")
    lines!(ax, idx, br.D_A2 ./ μ_A2; color = :steelblue,  label = "A Sc.2 (topo H₀)",
           linestyle = :dash)
    lines!(ax, 2:length(br.D_B)+1, br.D_B ./ μ_B;
           color = :darkorange, linewidth = 2, label = "B (phase-space H₁)")
    show_legend && axislegend(ax; position = :rt, framevisible = false, labelsize = 9)
end

function bearing_panel_m5!(ax, br; show_legend = false)
    bearing_panel!(ax, br; show_legend)
    for (m, col, off) in [(br.m_A1, :royalblue, -3),
                           (br.m_A2, :steelblue,   0),
                           (br.m_B,  :darkorange, +3)]
        d = m.significant && m.r_star / br.R > 0.6
        vlines!(ax, [m.r_star + off]; color = (:white, 0.5), linewidth = 5.0)
        vlines!(ax, [m.r_star + off];
                color     = (col, 1.0),
                linestyle = d ? :solid : :dot,
                linewidth = d ? 2.5 : 1.5)
    end
    show_legend && text!(ax, 0.02, 0.97;
        text  = "Vertical lines: Andrews r*\n(solid = significant, dotted = not)",
        align = (:left, :top), space = :relative,
        fontsize = 8, color = :gray40)
end

# ── 2.7  CROCKER matrix and plot ──────────────────────────────────────────────

function crocker_matrix(dgms::Vector{Vector{Tuple{Float64,Float64}}};
                         n_eps = 60)
    all_b = [b for d in dgms for (b, _) in d]
    all_d = [dd for d in dgms for (_, dd) in d]
    isempty(all_b) && return zeros(Int, n_eps, length(dgms)), range(0, 1; length = n_eps)
    ε_lo  = quantile(all_b, 0.05)
    ε_hi  = quantile(all_d, 0.95)
    εs    = range(ε_lo, ε_hi; length = n_eps)
    R     = length(dgms)
    C     = zeros(Int, n_eps, R)
    for r in 1:R
        isempty(dgms[r]) && continue
        for (j, ε) in enumerate(εs)
            C[j, r] = count(((b, d),) -> b ≤ ε ≤ d, dgms[r])
        end
    end
    C, εs
end

function load_B_dgms(ck::String)
    path = joinpath("checkpoints/nasa", ck * "_B_dgms.jls")
    isfile(path) || error("Checkpoint not found: $path — run §2 first")
    raw = deserialize(path)
    raw isa Vector{Vector{Tuple{Float64,Float64}}} && return raw
    # legacy DiagramCollection format
    [[(Float64(p[1]), Float64(p[2])) for p in (try d[2] catch; d end)
      if isfinite(Float64(p[2]))] for d in raw]
end

function crocker_panel!(ax, C, εs; title = "", r_star = nothing, clims = nothing)
    R      = size(C, 2)
    crange = isnothing(clims) ? (0, max(1, maximum(C))) : clims
    hm = heatmap!(ax, 1:R, collect(εs), C';
                  colormap = Reverse(:deep), colorrange = crange, interpolate = false)
    ax.title  = title
    ax.xlabel = "Trial r"
    ax.ylabel = "Filtration radius ε"
    isnothing(r_star) || vlines!(ax, [r_star]; color = :white, linewidth = 2.0,
                                  linestyle = :dash)
    hm
end

# ── 2.8  Run all bearings ─────────────────────────────────────────────────────

# Set 1
println("\n----- Set 1 (2156 trials) -----")
R1   = length(filter(!isdir, readdir(set1_dir)))
r0_1 = div(R1, 10)
b1_1 = nasa_analyse(load_channel(set1_dir, 1); r0 = r0_1,
                    label = "Bearing 1 — control",             ck = "set1_b1")
b1_3 = nasa_analyse(load_channel(set1_dir, 5); r0 = r0_1,
                    label = "Bearing 3 — inner race defect",   ck = "set1_b3")
b1_4 = nasa_analyse(load_channel(set1_dir, 7); r0 = r0_1,
                    label = "Bearing 4 — roller element defect", ck = "set1_b4")

# Set 2
println("\n----- Set 2 (984 trials) -----")
R2   = length(filter(!isdir, readdir(set2_dir)))
r0_2 = div(R2, 10)
b2_1 = nasa_analyse(load_channel(set2_dir, 1); r0 = r0_2,
                    label = "Bearing 1 — outer race failure",  ck = "set2_b1")

# Set 3 (optional)
b3_1 = b3_3 = nothing
if has_set3
    println("\n----- Set 3 (6324 trials) -----")
    R3   = length(filter(!isdir, readdir(set3_dir)))
    r0_3 = div(R3, 10)
    b3_1 = nasa_analyse(load_channel(set3_dir, 1); r0 = r0_3,
                        label = "Bearing 1 — control",             ck = "set3_b1")
    b3_3 = nasa_analyse(load_channel(set3_dir, 3); r0 = r0_3,
                        label = "Bearing 3 — outer race failure",  ck = "set3_b3")
end

# ── 2.9  Detection summary ────────────────────────────────────────────────────

println("\n=== Detection summary (Andrews sup-F, α=0.05, r*/R > 0.6) ===")
println("  Decision: significant AND r*/R > 0.6 → FAILURE")
verdict(m, R) = !m.significant ? "—" :
                (m.r_star / R > 0.6 ? "F (r*=$(m.r_star))" : "E (r*=$(m.r_star))")
@printf("  %-36s  %-18s %-18s %-18s\n", "Bearing", "A-Sc1", "A-Sc2", "B")
println("  " * "-"^92)
for br in filter(!isnothing, [b1_1, b1_3, b1_4, b2_1, b3_1, b3_3])
    @printf("  %-36s  %-18s %-18s %-18s\n",
            br.label,
            verdict(br.m_A1, br.R),
            verdict(br.m_A2, br.R),
            verdict(br.m_B,  br.R))
end

# ── 2.10  Distance sequence figures ──────────────────────────────────────────

# nasa_set1.pdf — raw normalised sequences
fig_s1 = Figure(size = (980, 980))
for (row, (br, ttl)) in enumerate([
        (b1_1, "Set 1 · Bearing 1 — control"),
        (b1_3, "Set 1 · Bearing 3 — inner race defect"),
        (b1_4, "Set 1 · Bearing 4 — roller element defect")])
    ax = Axis(fig_s1[row, 1]; title = ttl,
              xlabel = "Trial r", ylabel = "Distance / baseline mean")
    bearing_panel!(ax, br; show_legend = (row == 1))
end
save("plots/empirical/nasa_set1.pdf", fig_s1)
println("\nSaved → plots/empirical/nasa_set1.pdf")

# nasa_set2.pdf
fig_s2 = Figure(size = (980, 400))
ax = Axis(fig_s2[1, 1]; title = "Set 2 · Bearing 1 — outer race failure",
          xlabel = "Trial r", ylabel = "Distance / baseline mean")
bearing_panel!(ax, b2_1; show_legend = true)
save("plots/empirical/nasa_set2.pdf", fig_s2)
println("Saved → plots/empirical/nasa_set2.pdf")

# nasa_set1_m5.pdf — with change-point markers
fig_m1 = Figure(size = (980, 980))
for (row, (br, ttl)) in enumerate([
        (b1_1, "Set 1 · Bearing 1 — control"),
        (b1_3, "Set 1 · Bearing 3 — inner race defect"),
        (b1_4, "Set 1 · Bearing 4 — roller element defect")])
    ax = Axis(fig_m1[row, 1]; title = ttl,
              xlabel = "Trial r", ylabel = "Distance / baseline mean")
    bearing_panel_m5!(ax, br; show_legend = (row == 1))
end
save("plots/empirical/nasa_set1_m5.pdf", fig_m1)
println("Saved → plots/empirical/nasa_set1_m5.pdf")

# nasa_set2_m5.pdf
fig_m2 = Figure(size = (980, 480))
ax = Axis(fig_m2[1, 1]; title = "Set 2 · Bearing 1 — outer race failure",
          xlabel = "Trial r", ylabel = "Distance / baseline mean")
bearing_panel_m5!(ax, b2_1; show_legend = true)
save("plots/empirical/nasa_set2_m5.pdf", fig_m2)
println("Saved → plots/empirical/nasa_set2_m5.pdf")

# ── 2.11  CROCKER plots ───────────────────────────────────────────────────────

# Consensus r* values from detection analysis
R_STAR = Dict(
    "set1_b1" => nothing,
    "set1_b3" => b1_3.m_B.r_star,
    "set1_b4" => b1_4.m_B.r_star,
    "set2_b1" => b2_1.m_B.r_star,
)
if has_set3
    R_STAR["set3_b1"] = nothing
    R_STAR["set3_b3"] = b3_3.m_B.r_star
end

TITLES = Dict(
    "set1_b1" => "Set 1 · Bearing 1 — control",
    "set1_b3" => "Set 1 · Bearing 3 — inner race defect",
    "set1_b4" => "Set 1 · Bearing 4 — roller element defect",
    "set2_b1" => "Set 2 · Bearing 1 — outer race failure",
    "set3_b1" => "Set 3 · Bearing 1 — control",
    "set3_b3" => "Set 3 · Bearing 3 — outer race failure",
)

CROCKER = Dict{String,Tuple}()
for ck in collect(keys(R_STAR))
    dgms = load_B_dgms(ck)
    C, εs = cached(ck * "_crocker", "checkpoints/nasa") do
        println("  computing CROCKER matrix for $ck ($(length(dgms)) trials)...")
        crocker_matrix(dgms)
    end
    CROCKER[ck] = (C, εs)
    println("  $ck: β₁ range 0–$(maximum(C))")
end

# crocker_set1.pdf
fig_c1 = Figure(size = (1400, 1100))
cl1 = (0, maximum(maximum(CROCKER[k][1]) for k in ("set1_b1","set1_b3","set1_b4")))
for (row, ck) in enumerate(("set1_b1", "set1_b3", "set1_b4"))
    C, εs = CROCKER[ck]
    ax    = Axis(fig_c1[row, 1])
    hm    = crocker_panel!(ax, C, εs; title = TITLES[ck],
                           r_star = R_STAR[ck], clims = cl1)
    row == 1 && Colorbar(fig_c1[row, 2], hm; label = "β₁(ε, r)")
end
save("plots/empirical/crocker_set1.pdf", fig_c1)
println("Saved → plots/empirical/crocker_set1.pdf")

# crocker_set2.pdf
fig_c2 = Figure(size = (900, 500))
C2, ε2 = CROCKER["set2_b1"]
ax2    = Axis(fig_c2[1, 1])
hm2    = crocker_panel!(ax2, C2, ε2; title = TITLES["set2_b1"],
                        r_star = R_STAR["set2_b1"])
Colorbar(fig_c2[1, 2], hm2; label = "β₁(ε, r)")
save("plots/empirical/crocker_set2.pdf", fig_c2)
println("Saved → plots/empirical/crocker_set2.pdf")

# crocker_set3.pdf (only if Set 3 is available)
if has_set3
    fig_c3 = Figure(size = (1400, 750))
    cl3    = (0, maximum(maximum(CROCKER[k][1]) for k in ("set3_b1","set3_b3")))
    for (row, ck) in enumerate(("set3_b1", "set3_b3"))
        C, εs = CROCKER[ck]
        ax    = Axis(fig_c3[row, 1])
        hm    = crocker_panel!(ax, C, εs; title = TITLES[ck],
                               r_star = R_STAR[ck], clims = cl3)
        row == 1 && Colorbar(fig_c3[row, 2], hm; label = "β₁(ε, r)")
    end
    save("plots/empirical/crocker_set3.pdf", fig_c3)
    println("Saved → plots/empirical/crocker_set3.pdf")
end

##############################################################################
# §3  Victorian electricity demand forecasting
##############################################################################

println("\n" * "="^70)
println("§3  Victorian electricity demand forecasting")
println("="^70)

# ── 3.1  Download data ───────────────────────────────────────────────────────

const ELEC_URL  = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
const ELEC_PATH = "data/vic_elec.csv"
mkpath("data")

if isfile(ELEC_PATH)
    println("vic_elec already downloaded.")
else
    println("Downloading vic_elec from GitHub (tsibbledata)...")
    Downloads.download(ELEC_URL, ELEC_PATH)
    println("Downloaded → $ELEC_PATH")
end

# ── 3.2  Load and preprocess ─────────────────────────────────────────────────

df = CSV.read(ELEC_PATH, DataFrame)
println("Columns: ", names(df))

# Filter 2012–2013 (Excel serial dates: 40909 = 2012-01-01, 41639 = 2013-12-31)
filter!(r -> r.Date >= 40909 && r.Date <= 41639, df)
println("After date filter (2012–2013): $(nrow(df)) rows")

demand_col = "OperationalLessIndustrial"
y_raw = parse.(Float64, string.(df[!, demand_col]))
N_raw = length(y_raw)
println("N = $N_raw half-hourly observations")
@printf("Demand range: %.1f – %.1f MW\n", minimum(y_raw), maximum(y_raw))

# Weekly seasonal differencing: r_t = y_t - y_{t-336}
const S_ELEC = 336
y = y_raw[S_ELEC+1:end] .- y_raw[1:end-S_ELEC]
N = length(y)
@printf("After seasonal differencing: N = %d, σ = %.2f MW\n", N, std(y))

# ── 3.3  Parameters ───────────────────────────────────────────────────────────

const LAG_LIST = [1, 2, 48, 336]   # strong electricity baseline lags
const Q_ELEC   = length(LAG_LIST)
const W_ELEC   = 336               # topo window (1 week)
const K_ELEC   = 20                # refit every K steps
const N_TEST   = 2000
const H_LIST   = [1, 48]

τ_e = optimal_lag(y)
d_e = clamp(optimal_dim(y; lag = τ_e, max_dim = 6), 2, 6)
println("Embedding parameters: τ=$τ_e, dim=$d_e")

spec_e = TopoFeatureSpec(
    dim_max            = 1,
    dim                = d_e,
    lag                = τ_e,
    use_landscape      = true,
    n_landscape_layers = 2,
    n_landscape_grid   = 20,
    use_betti          = false,
    use_stats          = true,
    use_image          = false,
)
p_topo = length(feature_names(spec_e))
println("Topo feature dimension: $p_topo")

xgb_params = (num_round = 150, max_depth = 5, eta = 0.05,
               subsample = 0.8, colsample_bytree = 0.8,
               objective = "reg:squarederror", verbosity = 0)

# ── 3.4  Topo feature matrix ──────────────────────────────────────────────────

n_windows  = N - W_ELEC + 1
ckpt_topo  = "checkpoints/elec/F_topo.jls"

if isfile(ckpt_topo)
    println("Loading topo features from checkpoint...")
    F_topo, tgrid_land = deserialize(ckpt_topo)
    println("Loaded: $(size(F_topo))")
else
    println("Computing topo features ($n_windows windows, may take ~30 min)...")
    _dgms0 = persistent_homology(
                 embed(y[1:W_ELEC]; dim = spec_e.dim, lag = spec_e.lag);
                 dim_max = 1)
    tgrid_land = landscape(_dgms0, 0;
                     n_grid   = spec_e.n_landscape_grid,
                     n_layers = spec_e.n_landscape_layers).tgrid
    F_topo = Matrix{Float64}(undef, n_windows, p_topo)
    for i in 1:n_windows
        i % 5_000 == 1 && @printf("  window %d/%d\n", i, n_windows)
        F_topo[i, :] = topo_features(y[i:i+W_ELEC-1]; spec = spec_e,
                                      tgrid_landscape = tgrid_land,
                                      tgrid_betti     = nothing)
    end
    F_topo[isnan.(F_topo) .| isinf.(F_topo)] .= 0.0
    serialize(ckpt_topo, (F_topo, tgrid_land))
    println("Done. Checkpointed.")
end

# ── 3.5  Walk-forward evaluation ──────────────────────────────────────────────

σ_e   = std(y)
i_min = max(1, maximum(LAG_LIST) - W_ELEC + 2)
make_lag_row(i) = [y[i+W_ELEC-1-l] for l in LAG_LIST]'

elec_results = Dict{Int, NamedTuple}()

for h in H_LIST
    n_valid    = n_windows - h
    test_start = max(n_valid - N_TEST + 1, maximum(LAG_LIST) + 1)
    n_test_act = n_valid - test_start + 1

    println("\n--- h=$h ($(h*30) min ahead) | test: $n_test_act points ---")

    ae_base = Float64[]; ae_aug = Float64[]
    bst_b   = nothing;   bst_a  = nothing

    for t_test in test_start:n_valid
        if isnothing(bst_b) || (t_test - test_start) % K_ELEC == 0
            X_lag_tr = reduce(vcat, make_lag_row(i) for i in i_min:t_test-1)
            X_top_tr = F_topo[i_min:t_test-1, :]
            y_tr     = [y[i+W_ELEC-1+h] for i in i_min:t_test-1]
            bst_b    = xgboost((X_lag_tr, y_tr); xgb_params...)
            bst_a    = xgboost((hcat(X_lag_tr, X_top_tr), y_tr); xgb_params...)
        end
        X_lag_te = make_lag_row(t_test)
        X_aug_te = hcat(X_lag_te, F_topo[t_test:t_test, :])
        y_te     = y[t_test+W_ELEC-1+h]
        push!(ae_base, abs(XGBoost.predict(bst_b, X_lag_te)[1] - y_te))
        push!(ae_aug,  abs(XGBoost.predict(bst_a, X_aug_te)[1] - y_te))
    end

    mae_b   = mean(ae_base) / σ_e;  mae_a  = mean(ae_aug)  / σ_e
    rmse_b  = sqrt(mean(ae_base .^ 2)) / σ_e
    rmse_a  = sqrt(mean(ae_aug  .^ 2)) / σ_e
    Δmae    = (mae_b - mae_a) / mae_b * 100
    Δrmse   = (rmse_b - rmse_a) / rmse_b * 100
    elec_results[h] = (; mae_b, mae_a, rmse_b, rmse_a, Δmae, Δrmse, n = n_test_act)

    @printf("  Baseline   MAE=%.4f  RMSE=%.4f\n", mae_b, rmse_b)
    @printf("  Augmented  MAE=%.4f  RMSE=%.4f\n", mae_a, rmse_a)
    @printf("  Δ_MAE=%+.2f%%  Δ_RMSE=%+.2f%%\n", Δmae, Δrmse)
end

# ── 3.6  Save results ─────────────────────────────────────────────────────────

serialize("checkpoints/elec/results.jls", elec_results)
CSV.write("results/elec_forecasting.csv",
    DataFrame([(h = h,
                mae_baseline  = elec_results[h].mae_b,
                mae_augmented = elec_results[h].mae_a,
                delta_mae_pct = elec_results[h].Δmae,
                rmse_baseline = elec_results[h].rmse_b,
                rmse_augmented= elec_results[h].rmse_a,
                delta_rmse_pct= elec_results[h].Δrmse,
                n_test        = elec_results[h].n)
               for h in H_LIST]))

println("\n" * "="^70)
println("FORECASTING SUMMARY  (σ=$(round(σ_e, digits=1)) MW, normalised)")
println("="^70)
@printf("%-4s  %-12s  %7s  %7s  %7s  %7s\n",
        "h", "Model", "MAE", "RMSE", "ΔMAE%", "ΔRMSE%")
println("-"^54)
for h in H_LIST
    r = elec_results[h]
    @printf("%-4d  %-12s  %7.4f  %7.4f\n",          h, "Baseline",  r.mae_b, r.rmse_b)
    @printf("%-4s  %-12s  %7.4f  %7.4f  %+6.2f%%  %+6.2f%%\n",
            "",   "Augmented", r.mae_a, r.rmse_a, r.Δmae, r.Δrmse)
    println("-"^54)
end
println("Results saved → results/elec_forecasting.csv")

##############################################################################
# Summary
##############################################################################

println("""
\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empirical demo complete.  Output files:

  plots/empirical/
    nasa_set1.pdf          — Set 1 distance sequences
    nasa_set2.pdf          — Set 2 distance sequences
    nasa_set1_m5.pdf       — Set 1 with Andrews r* markers
    nasa_set2_m5.pdf       — Set 2 with Andrews r* markers
    crocker_set1.pdf       — CROCKER plots, Set 1
    crocker_set2.pdf       — CROCKER plot,  Set 2
    crocker_set3.pdf       — CROCKER plots, Set 3 (if data present)

  results/
    elec_forecasting.csv   — MAE/RMSE for h=1 and h=48

  checkpoints/             — intermediate results for re-runs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
