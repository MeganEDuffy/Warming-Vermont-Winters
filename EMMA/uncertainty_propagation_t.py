"""
uncertainty_propagation.py
Propagates uncertainty for EMMA mixing fractions using numerical differentiation 
and Student's t-statistics, similar to the framework of Genereux (1998).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t


def solve_fractions(C_mixture, C_endmembers):
    """
    Core linear solver with constraints sum(f) == 1 and 0 <= f <= 1.
    C_mixture: 1D array of tracer values for one stream sample (n_tracers,)
    C_endmembers: 2D array of endmember tracer values (n_sources, n_tracers)
    """
    n_sources = C_endmembers.shape[0]

    def objective(f):
        pred = np.dot(f, C_endmembers)
        return np.linalg.norm(C_mixture - pred)

    constraints = (
        {"type": "eq", "fun": lambda f: np.sum(f) - 1},
        {"type": "ineq", "fun": lambda f: f},
        {"type": "ineq", "fun": lambda f: 1 - f},
    )

    init_guess = np.ones(n_sources) / n_sources
    res = minimize(
        objective, init_guess, constraints=constraints, method="SLSQP"
    )
    return res.x if res.success else np.full(n_sources, np.nan)


def propagate_genereux_uncertainty(
    stream_df,
    em_grouped,
    em_raw,
    tracers,
    analytical_sd=None,
    confidence_level=0.70,
    epsilon=1e-5,
):
    """
    Calculates absolute uncertainty for each sample's source fractions 
    incorporating Student's t-statistics per Genereux (1998).

    Parameters:
        stream_df (DataFrame): Streamwater samples (already subsetted for the event).
        em_grouped (DataFrame): Mean tracer values for each end-member Type.
        em_raw (DataFrame): Raw end-member samples (used to compute sample SD and n).
        tracers (list): List of tracers to use in calculations.
        analytical_sd (dict): Analytical standard deviations for stream tracers.
        confidence_level (float): Target confidence level (default 0.70).
        epsilon (float): Step size for numerical partial derivatives.

    Returns:
        uncertainties_df (DataFrame): Table with sample IDs, datetimes, and uncertainty 
                                      for each source.
    """
    # Clean streamwater of NaNs
    stream_clean = stream_df.dropna(subset=tracers).copy()

    # Identify sources and extract their means
    sources = em_grouped["Type"].values
    n_sources = len(sources)
    n_tracers = len(tracers)

    # 1. Compute Sample Sizes (n) and Standard Deviations (s) for end-members
    em_counts = em_raw.groupby("Type")[tracers].count()
    em_sd = em_raw.groupby("Type")[tracers].std()

    # Alpha for two-tailed t-distribution (e.g., 1 - 0.70 = 0.30)
    alpha = 1.0 - confidence_level

    # Build an end-member error matrix incorporating Student's t-statistics
    # Genereux framework: Error of the mean component involves t * (s / sqrt(n)) 
    # or t * assumed_error for n=1.
    adjusted_em_sd = np.zeros((n_sources, n_tracers))

    for i, source in enumerate(sources):
        for j, col in enumerate(tracers):
            n_samples = em_counts.loc[source, col] if source in em_counts.index else 0
            sample_sd = em_sd.loc[source, col] if source in em_sd.index else np.nan

            if pd.isna(sample_sd) or n_samples <= 1:
                # Fallback for n = 1 (e.g., coefficient of variation of the mean)
                mean_val = em_grouped.loc[em_grouped["Type"] == source, col].values[0]
                fallback_s = abs(mean_val) * 0.39  # coefficient of variation derived from RI25 end-member samples. 
                # Grouped CoV derived from analysis of baseflow, snmowlet, and soil water samples taken on consecutive days.
                # See notebooks on Wade and Hungerford end-member tracer variation.
                
                # For n=1, degrees of freedom = 1 (or small-sample proxy). 
                # Using df=1 with two-tailed alpha=0.30 gives t ≈ 1.963
                df = 1 
                t_val = t.ppf(1.0 - alpha / 2.0, df=df)
                
                # Total uncertainty of the mean representation
                adjusted_em_sd[i, j] = t_val * fallback_s
            else:
                # Standard Student's t adjustment for n > 1
                df = n_samples - 1
                t_val = t.ppf(1.0 - alpha / 2.0, df=df)
                
                # Standard error of the mean scaled by t
                se_mean = sample_sd / np.sqrt(n_samples)
                adjusted_em_sd[i, j] = t_val * se_mean

    # Streamwater analytical standard deviations
    if analytical_sd is None:
        analytical_sd = {t: stream_clean[t].mean() * 0.05 for t in tracers}

    # Extract clean arrays
    em_means_matrix = em_grouped.set_index("Type")[tracers].loc[sources].values

    # Pre-allocate output arrays
    uncertainty_results = []

    # 2. Iterate through each stream sample
    for idx, row in stream_clean.iterrows():
        C_m = row[tracers].values.astype(float)

        # Baseline calculated fractions
        f_base = solve_fractions(C_m, em_means_matrix)

        # Variance accumulations for each source fraction [Var_f1, Var_f2, ...]
        variance_accum = np.zeros(n_sources)

        # --- PART A: Derivatives w.r.t stream mxture tracers (C_m) ---
        # Note: Stream analytical errors can also be scaled by normal/t distributions, 
        # but stream measurement replicates often use standard analytical precision (1-sigma ~ normal).
        for j, tracer in enumerate(tracers):
            s_m = analytical_sd[tracer]

            C_m_up = C_m.copy()
            C_m_up[j] += epsilon
            f_up = solve_fractions(C_m_up, em_means_matrix)

            C_m_down = C_m.copy()
            C_m_down[j] -= epsilon
            f_down = solve_fractions(C_m_down, em_means_matrix)

            df_dCm = (f_up - f_down) / (2 * epsilon)
            variance_accum += (df_dCm * s_m) ** 2

        # --- PART B: Derivatives w.r.t Endmember Concentrations (C_em) ---
        for i, source in enumerate(sources):
            for j in range(n_tracers):
                s_em = adjusted_em_sd[i, j]

                em_up = em_means_matrix.copy()
                em_up[i, j] += epsilon
                f_up = solve_fractions(C_m, em_up)

                em_down = em_means_matrix.copy()
                em_down[i, j] -= epsilon
                f_down = solve_fractions(C_m, em_down)

                df_dec = (f_up - f_down) / (2 * epsilon)
                variance_accum += (df_dec * s_em) ** 2

        # Total uncertainty at the specified confidence interval
        W_f = np.sqrt(variance_accum)
        uncertainty_results.append(W_f)

    # 3. Assemble output dataFrame
    uncertainties_matrix = np.vstack(uncertainty_results)
    cols = [f"{source}_Uncertainty_{int(confidence_level*100)}sig" for source in sources]

    out_df = pd.DataFrame(uncertainties_matrix, columns=cols)
    out_df.insert(0, "Sample ID", stream_clean["Sample ID"].values)
    out_df.insert(1, "Datetime", stream_clean["Datetime"].values)

    return out_df