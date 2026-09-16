"""
uncertainty_propagation.py
Propagates 1-sigma uncertainty for EMMA mixing fractions using numerical 
differentiation, similar to framework of Genereux (1998).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    epsilon=1e-5,
):
    """
    Calculates 1-sigma uncertainty for each sample's source fractions.

    Parameters:
        stream_df (DataFrame): Streamwater samples (already subsetted for the event).
        em_grouped (DataFrame): Mean tracer values for each end-member Type.
        em_raw (DataFrame): Raw end-member samples (used to compute sample SD).
        tracers (list): List of tracers to use in calculations.
        analytical_sd (dict): Dictionary of analytical standard deviations for
        each tracer.
                              If none, defaults to 0.05 * mean stream value.
        epsilon (float): Step size for numerical partial derivatives.

    Returns:
        uncertainties_df (DataFrame): Table with sample IDs, datetimes, and 1-sigma
                                      uncertainty (W_f) for each source.
    """
    # Clean streamwater of NaNs
    stream_clean = stream_df.dropna(subset=tracers).copy()

    # Identify sources and extract their means
    sources = em_grouped["Type"].values
    n_sources = len(sources)
    n_tracers = len(tracers)

    # 1. Estimate Tracer Standard Deviations (s_C)
    # Endmember standard deviations (spatial/temporal variation)
    em_sd = em_raw.groupby("Type")[tracers].std()
    # If an endmember has only 1 sample (SD is NaN), assign an assumed (literature) uncertainty
    # (e.g., 10% coefficient of variation) so the model doesn't treat it as 0% error.
    for col in tracers:
        fallback_sd = em_raw.groupby("Type")[col].mean() * 0.1 #10% CoV
        em_sd[col] = em_sd[col].fillna(fallback_sd)

    # Streamwater analytical standard deviations
    if analytical_sd is None:
        # Default fallback: 5% of the overall streamwater mean for each tracer
        analytical_sd = {t: stream_clean[t].mean() * 0.05 for t in tracers}

    # Extract clean arrays
    em_means_matrix = em_grouped.set_index("Type")[tracers].loc[sources].values
    em_sd_matrix = em_sd.loc[sources].values

    # Pre-allocate output arrays
    uncertainty_results = []

    # 2. Iterate through each stream sample
    for idx, row in stream_clean.iterrows():
        C_m = row[tracers].values.astype(float)

        # Baseline calculated fractions
        f_base = solve_fractions(C_m, em_means_matrix)

        # Variance accumulations for each source fraction [Var_f1, Var_f2, ...]
        variance_accum = np.zeros(n_sources)

        # --- PART A: Derivatives w.r.t Stream Mixture Tracers (C_m) ---
        for j, tracer in enumerate(tracers):
            s_m = analytical_sd[tracer]

            # Perturb up
            C_m_up = C_m.copy()
            C_m_up[j] += epsilon
            f_up = solve_fractions(C_m_up, em_means_matrix)

            # Perturb down
            C_m_down = C_m.copy()
            C_m_down[j] -= epsilon
            f_down = solve_fractions(C_m_down, em_means_matrix)

            # Numerical derivative df/dC_m
            df_dCm = (f_up - f_down) / (2 * epsilon)

            # Accumulate variance contribution: (df/dCm * s_m)^2
            variance_accum += (df_dCm * s_m) ** 2

        # --- PART B: Derivatives w.r.t Endmember Concentrations (C_em) ---
        for i, source in enumerate(sources):
            for j in range(n_tracers):
                s_em = em_sd_matrix[i, j]

                # Perturb the specific end-member tracer coordinate up
                em_up = em_means_matrix.copy()
                em_up[i, j] += epsilon
                f_up = solve_fractions(C_m, em_up)

                # Perturb down
                em_down = em_means_matrix.copy()
                em_down[i, j] -= epsilon
                f_down = solve_fractions(C_m, em_down)

                # Numerical derivative df/dC_em
                df_dec = (f_up - f_down) / (2 * epsilon)

                # Accumulate variance contribution: (df/dC_em * s_em)^2
                variance_accum += (df_dec * s_em) ** 2

        # 1-sigma absolute uncertainty is the square root of total variance
        W_f = np.sqrt(variance_accum)
        uncertainty_results.append(W_f)

    # 3. Assemble Output DataFrame
    uncertainties_matrix = np.vstack(uncertainty_results)
    cols = [f"{source}_Uncertainty_1sig" for source in sources]

    out_df = pd.DataFrame(uncertainties_matrix, columns=cols)
    out_df.insert(0, "Sample ID", stream_clean["Sample ID"].values)
    out_df.insert(1, "Datetime", stream_clean["Datetime"].values)

    return out_df