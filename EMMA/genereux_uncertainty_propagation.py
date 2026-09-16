"""
genereux_uncertainty_propagation.py

Propagates uncertainty for 3-component EMMA mixing fractions using 
closed-form partial derivatives and Student's t-statistics as in Genereux (2022), addendum to Genereux (1998).

Adapted from 2 tracers to the first 2 principle components.
"""

import numpy as np
import pandas as pd
from scipy.stats import t


def compute_genereux2022_fractions_and_derivatives(A_S, B_S, A_1, B_1, A_2, B_2, A_3, B_3):
    """
    Computes 3-component mixing fractions (f1, f2, f3) and all 24 partial 
    derivatives using analytical formulations from Genereux (2022).
    A = PC1 score, B = PC2 score.
    """
    M1 = A_S*B_2 - A_S*B_3 + A_2*B_3 - A_2*B_S + A_3*B_S - A_3*B_2
    M2 = A_S*B_3 - A_S*B_1 + A_1*B_S - A_1*B_3 + A_3*B_1 - A_3*B_S
    M3 = A_S*B_1 - A_S*B_2 + A_1*B_2 - A_1*B_S + A_2*B_S - A_2*B_1
    M4 = A_1*B_2 - A_1*B_3 + A_2*B_3 - A_2*B_1 + A_3*B_1 - A_3*B_2

    f1, f2, f3 = M1 / M4, M2 / M4, M3 / M4
    M4_sq = M4**2

    # 24 Partial Derivatives (Genereux 2022 Eq. 5 - 28)
    df1_dAS = (B_2 - B_3) / M4
    df1_dA1 = -M1 * (B_2 - B_3) / M4_sq
    df1_dA2 = (M4 * (B_3 - B_S) - M1 * (B_3 - B_1)) / M4_sq
    df1_dA3 = (M4 * (B_S - B_2) - M1 * (B_1 - B_2)) / M4_sq

    df1_dBS = (A_3 - A_2) / M4
    df1_dB1 = -M1 * (A_3 - A_2) / M4_sq
    df1_dB2 = (M4 * (A_S - A_3) - M1 * (A_1 - A_3)) / M4_sq
    df1_dB3 = (M4 * (A_2 - A_S) - M1 * (A_2 - A_1)) / M4_sq

    df2_dAS = (B_3 - B_1) / M4
    df2_dA1 = (M4 * (B_S - B_3) - M2 * (B_2 - B_3)) / M4_sq
    df2_dA2 = -M2 * (B_3 - B_1) / M4_sq
    df2_dA3 = (M4 * (B_1 - B_S) - M2 * (B_1 - B_2)) / M4_sq

    df2_dBS = (A_1 - A_3) / M4
    df2_dB1 = (M4 * (A_3 - A_S) - M2 * (A_3 - A_2)) / M4_sq
    df2_dB2 = -M2 * (A_1 - A_3) / M4_sq
    df2_dB3 = (M4 * (A_S - A_1) - M2 * (A_2 - A_1)) / M4_sq

    df3_dAS = (B_1 - B_2) / M4
    df3_dA1 = (M4 * (B_2 - B_S) - M3 * (B_2 - B_3)) / M4_sq
    df3_dA2 = (M4 * (B_S - B_1) - M3 * (B_3 - B_1)) / M4_sq
    df3_dA3 = -M3 * (B_1 - B_2) / M4_sq

    df3_dBS = (A_2 - A_1) / M4
    df3_dB1 = (M4 * (A_S - A_2) - M3 * (A_3 - A_2)) / M4_sq
    df3_dB2 = (M4 * (A_1 - A_S) - M3 * (A_1 - A_3)) / M4_sq
    df3_dB3 = -M3 * (A_2 - A_1) / M4_sq

    fractions = (f1, f2, f3)
    derivatives = {
        'f1': [df1_dAS, df1_dA1, df1_dA2, df1_dA3, df1_dBS, df1_dB1, df1_dB2, df1_dB3],
        'f2': [df2_dAS, df2_dA1, df2_dA2, df2_dA3, df2_dBS, df2_dB1, df2_dB2, df2_dB3],
        'f3': [df3_dAS, df3_dA1, df3_dA2, df3_dA3, df3_dBS, df3_dB1, df3_dB2, df3_dB3]
    }
    return fractions, derivatives


def propagate_genereux2022_pca_uncertainty(
    stream_df,
    em_grouped_pca,
    em_raw,
    tracers,
    scaler,
    pca,
    pc_analytical_sd={'PC1': 0.05, 'PC2': 0.05},
    confidence_level=0.70,
):
    # Extract exact tracer order expected by fitted scaler
    if hasattr(scaler, "feature_names_in_"):
        ordered_tracers = list(scaler.feature_names_in_)
    else:
        ordered_tracers = tracers

    # Ensure em_grouped_pca has PC1 and PC2 columns
    em_grouped_clean = em_grouped_pca.copy()
    col_map = {str(c).upper(): c for c in em_grouped_clean.columns}

    if "PC1" in col_map and "PC2" in col_map:
        em_grouped_clean = em_grouped_clean.rename(
            columns={col_map["PC1"]: "PC1", col_map["PC2"]: "PC2"}
        )
    elif "COMP1" in col_map and "COMP2" in col_map:
        em_grouped_clean = em_grouped_clean.rename(
            columns={col_map["COMP1"]: "PC1", col_map["COMP2"]: "PC2"}
        )
    else:
        # Transform end-member mean concentrations into PC-space
        em_scaled = scaler.transform(em_grouped_clean[ordered_tracers])
        em_scores = pca.transform(em_scaled)[:, :2]
        em_grouped_clean["PC1"] = em_scores[:, 0]
        em_grouped_clean["PC2"] = em_scores[:, 1]

    sources = em_grouped_clean["Type"].values
    if len(sources) != 3:
        raise ValueError("Genereux (2022) equations require exactly 3 end-members.")

    # 1. Transform raw end-members into PC-space to calculate SD and n
    em_raw_clean = em_raw.dropna(subset=ordered_tracers).copy()
    em_raw_scaled = scaler.transform(em_raw_clean[ordered_tracers])
    em_raw_scores = pca.transform(em_raw_scaled)[:, :2]

    em_raw_clean["PC1"] = em_raw_scores[:, 0]
    em_raw_clean["PC2"] = em_raw_scores[:, 1]

    em_counts = em_raw_clean.groupby("Type")[["PC1", "PC2"]].count()
    em_sd = em_raw_clean.groupby("Type")[["PC1", "PC2"]].std()

    alpha = 1.0 - confidence_level

    # Compute t-adjusted uncertainties in PC space
    W_em = {}
    for source in sources:
        for pc_col in ["PC1", "PC2"]:
            n_samples = em_counts.loc[source, pc_col] if source in em_counts.index else 0
            sample_sd = em_sd.loc[source, pc_col] if source in em_sd.index else np.nan

            if pd.isna(sample_sd) or n_samples <= 1:
                mean_val = em_grouped_clean.loc[em_grouped_clean["Type"] == source, pc_col].values[0]
                fallback_s = max(abs(mean_val) * 0.10, 0.05)
                t_val = t.ppf(1.0 - alpha / 2.0, df=1)
                W_em[(source, pc_col)] = t_val * fallback_s
            else:
                df_deg = n_samples - 1
                t_val = t.ppf(1.0 - alpha / 2.0, df=df_deg)
                se_mean = sample_sd / np.sqrt(n_samples)
                W_em[(source, pc_col)] = t_val * se_mean

    # 2. Transform stream samples into PC-space
    stream_clean = stream_df.dropna(subset=ordered_tracers).copy()
    stream_scaled = scaler.transform(stream_clean[ordered_tracers])
    stream_scores = pca.transform(stream_scaled)[:, :2]

    stream_clean["PC1"] = stream_scores[:, 0]
    stream_clean["PC2"] = stream_scores[:, 1]

    # End-member mean PC scores
    em_dict = em_grouped_clean.set_index("Type")
    A_1, B_1 = em_dict.loc[sources[0], "PC1"], em_dict.loc[sources[0], "PC2"]
    A_2, B_2 = em_dict.loc[sources[1], "PC1"], em_dict.loc[sources[1], "PC2"]
    A_3, B_3 = em_dict.loc[sources[2], "PC1"], em_dict.loc[sources[2], "PC2"]

    W_AS, W_BS = pc_analytical_sd["PC1"], pc_analytical_sd["PC2"]
    W_A1, W_B1 = W_em[(sources[0], "PC1")], W_em[(sources[0], "PC2")]
    W_A2, W_B2 = W_em[(sources[1], "PC1")], W_em[(sources[1], "PC2")]
    W_A3, W_B3 = W_em[(sources[2], "PC1")], W_em[(sources[2], "PC2")]

    W_vector = np.array([W_AS, W_A1, W_A2, W_A3, W_BS, W_B1, W_B2, W_B3])

    results = []

    # 3. Compute fractions and propagate error per sample
    for idx, row in stream_clean.iterrows():
        A_S = float(row["PC1"])
        B_S = float(row["PC2"])

        fractions, derivs = compute_genereux2022_fractions_and_derivatives(
            A_S, B_S, A_1, B_1, A_2, B_2, A_3, B_3
        )

        W_f1 = np.sqrt(np.sum((np.array(derivs['f1']) * W_vector) ** 2))
        W_f2 = np.sqrt(np.sum((np.array(derivs['f2']) * W_vector) ** 2))
        W_f3 = np.sqrt(np.sum((np.array(derivs['f3']) * W_vector) ** 2))

        results.append({
            "Sample ID": row["Sample ID"],
            "Datetime": row["Datetime"],
            f"{sources[0]}_f": fractions[0],
            f"{sources[1]}_f": fractions[1],
            f"{sources[2]}_f": fractions[2],
            f"{sources[0]}_Uncertainty_{int(confidence_level*100)}sig": W_f1,
            f"{sources[1]}_Uncertainty_{int(confidence_level*100)}sig": W_f2,
            f"{sources[2]}_Uncertainty_{int(confidence_level*100)}sig": W_f3,
        })

    return pd.DataFrame(results)

def calculate_pc_analytical_sd(scaler, pca, analytical_sd_dict):
    """
    Propagates individual tracer analytical precisions into PC1 and PC2 space.
    """
    ordered_tracers = list(scaler.feature_names_in_)
    sigma = scaler.scale_
    loadings = pca.components_  # shape (n_components, n_tracers)
    
    # Vector of laboratory precisions aligned with tracer order
    w_c = np.array([analytical_sd_dict[t] for t in ordered_tracers])
    
    w_pc1 = np.sqrt(np.sum(((loadings[0, :] / sigma) * w_c) ** 2))
    w_pc2 = np.sqrt(np.sum(((loadings[1, :] / sigma) * w_c) ** 2))
    
    return {'PC1': w_pc1, 'PC2': w_pc2}