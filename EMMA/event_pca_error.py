############################################
# Python module to perform single evnt PCA #
# Megan Duffy - Adair Lab, UVM #############
# last updated 2026-01-21 ##################
############################################

# Event_pca_error.py

# Tracers are set up for Wade, Hungerford, and Potash Brooks
# Error bars are +/- 1 STDV given multiple endmembers of the same type (i.e., three snowmelt lysimeter samples)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

from scipy.stats import t


def plot_event_pca_with_avgEM(
    data,
    site,
    start_date,
    end_date,
    endmember_ids,
    title="Event-Specific PCA"
):
    """
    Generate PCA plot for a specific storm event, following the EMMALAB workflow:
    PCA is fit ONLY on streamwater (mixture) data, then endmembers are projected
    into that PCA space.
    Also plots AvgEMScore: mean PC score for each endmember type.
    """

    # Site-specific tracers
    if site == "Wade":
        #tracers = ['Ca_mg_L', 'Si_mg_L', 'Mg_mg_L', 'dD', 'd18O', 'Na_mg_L'] # Original Jan '26 WRR submission tracer selection
        tracers = ['Ca_mg_L', 'Na_mg_L', 'Mg_mg_L']                           # Aug '26 WRR resubmission traacer selection after more thourough vetting
    elif site == "Hungerford":
        #tracers = ['Ca_mg_L', 'Cl_mg_L', 'Si_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O'] # Original Jan '26 WRR submission tracer selection
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'Cu_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']  # Aug '26 WRR resubmission traacer selection after more thourough vetting  Ca, Cl, Cu, d18O, dD, K, Mg, Na
    elif site == "Potash":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    else:
        raise ValueError("Site not recognized. Use 'Wade', 'Potash', or 'Hungerford'.")

    # Ensure datetime column is datetime type
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    # Subset streamwater (mixture) in date range and site
    stream = data[
        (data["Site"] == site) &
        (data["Type"].isin(["Grab", "Grab/Isco", "Baseflow", "Isco"])) &
        (data["Date"] >= pd.to_datetime(start_date)) &
        (data["Date"] <= pd.to_datetime(end_date))
    ].copy()

    # Subset endmembers by Sample ID
    endmembers = data[
        (data["Site"] == site) &
        (data["Sample ID"].isin(endmember_ids))
    ].copy()

    # Drop NA for stream tracers
    subset_stream = stream[tracers].dropna().copy()
    subset_stream["Group"] = "Streamwater"
    subset_stream["Type"] = "Streamwater"
    subset_stream["Date"] = stream["Date"]

    # Fill NA for endmembers with mean (per tracer)
    subset_endmembers = endmembers[tracers].copy()
    subset_endmembers = subset_endmembers.fillna(subset_endmembers.mean())
    subset_endmembers["Group"] = "Endmember"
    subset_endmembers["Type"] = endmembers["Type"].values
    subset_endmembers["Date"] = endmembers["Date"].values

    # -----------------------
    # EMMALAB PCA logic
    # -----------------------
    scaler = StandardScaler()
    scaled_stream = scaler.fit_transform(subset_stream[tracers])

    pca = PCA(n_components=2)
    stream_pca_result = pca.fit_transform(scaled_stream)
    subset_stream["PC1"] = stream_pca_result[:, 0]
    subset_stream["PC2"] = stream_pca_result[:, 1]

    scaled_endmembers = scaler.transform(subset_endmembers[tracers])
    endmember_pca_result = pca.transform(scaled_endmembers)
    subset_endmembers["PC1"] = endmember_pca_result[:, 0]
    subset_endmembers["PC2"] = endmember_pca_result[:, 1]

    # Combine
    combined = pd.concat([subset_stream, subset_endmembers], ignore_index=True)

    # -----------------------
    # Compute AvgEMScore + SD
    # -----------------------
    stats_em = (
        subset_endmembers.groupby("Type")[["PC1", "PC2"]]
        .agg(['mean', 'std'])
    )
    stats_em.columns = ['PC1_mean', 'PC1_std', 'PC2_mean', 'PC2_std']
    stats_em = stats_em.reset_index()
    stats_em["Group"] = "AvgEMScore"

    # -----------------------
    # Plotting
    # -----------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # --- set font sizes globally (scale ~1.5× default) ---
    mpl.rcParams.update({
        "font.size": 30,          # base font size
        "axes.titlesize": 28,     # title
        "axes.labelsize": 28,     # x and y labels
        "xtick.labelsize": 25,    # tick labels
        "ytick.labelsize": 25,
        "legend.fontsize": 25
    })

    # Streamwater points
    sw = combined[combined["Group"] == "Streamwater"]
    ax.scatter(sw["PC1"], sw["PC2"], marker='o', s=110, c='none', edgecolors='royalblue', linewidth = 2, alpha=1, label='Streamwater')

    # Endmember markers/colors
    endmember_markers = {
        #'Rain': 'o', 'Snow': 's', 
        'Soil water lysimeter': '<', #'Soil water lysimeter wet': '>', # I removed distinction between wet and dry transects 2026-07-27
        'Snowmelt lysimeter': '^', 
        #'Precip': '*',
        'Groundwater': '>', 'Baseflow': 'P'
    }
    colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba',
              '#2ca25f', '#636363', '#8856a7', '#d95f0e']

    # Plot mean + error bars (instead of individual endmembers)
    for (etype, color) in zip(endmember_markers.keys(), colors):
        em_stat = stats_em[stats_em["Type"] == etype]
        if not em_stat.empty:
            ax.errorbar(
                em_stat["PC1_mean"], em_stat["PC2_mean"],
                xerr=em_stat["PC1_std"], yerr=em_stat["PC2_std"],
                fmt=endmember_markers[etype],  # marker style
                color=color, ecolor="black",
                elinewidth=1.5, capsize=4,
                markersize=15, markeredgecolor='black',
                label=f"{etype}"
            )

    # Draw mixing space polygon
    if len(stats_em) >= 3:
        from scipy.spatial import ConvexHull
        import numpy as np

        points = stats_em[["PC1_mean", "PC2_mean"]].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.vstack([hull_points, hull_points[0]])

        ax.plot(hull_points[:,0], hull_points[:,1],
                linestyle='-', color='black', linewidth=1.5,
                label="Mixing space")
        ax.fill(hull_points[:,0], hull_points[:,1],
                facecolor='grey', alpha=0.1)

    # Variance explained
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")

    ax.set_title(title)

    # --- Save plot as 'output/site_title.jpg' ---
    # Sanitize filename: replace spaces and parentheses
    save_event_name = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    filename = f"{site}_{save_event_name}.jpg"
    output_path = os.path.join("/home/millieginty/OneDrive/git-repos/LCBP-interannual-EMMAs/Output/PCA-Mixing", filename)

    #ax.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left") # Uncomment to get legend and manually input into composite figure
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to output dir")
    

    plt.show()

def calculate_pc_uncertainty(
    etype_data, 
    tracers, 
    scaler, 
    pca, 
    analytical_dict
):
    """
    Propagates tracer uncertainty into PC space for a specific end-member type.
    Uses Genereux (1998) logic + PCA loadings.
    """
    # 1. Calculate Standard Deviation (s) of the end-member samples
    # If n=1, standard deviation is 0, and we rely on analytical uncertainty later
    s = etype_data[tracers].std().fillna(0)
    n = len(etype_data)
    
    # 2. Get Student's t-value for 95% confidence (Genereux Eq 4)
    # Using n-1 degrees of freedom; for n=1, we'll default to a high value or analytical
    t_val = t.ppf(0.975, max(n-1, 1)) if n > 1 else 1.0 

    # 3. Calculate W for each raw tracer (Eq 4 in Genereux)
    # If you have a 'literature' estimate, you'd add it here as a third term
    W_tracers = []
    for tracer in tracers:
        Wa = analytical_dict.get(tracer, 0)
        # Combined uncertainty: sqrt((t*s/sqrt(n))^2 + Wa^2)
        W = np.sqrt((t_val * s[tracer] / np.sqrt(n))**2 + Wa**2)
        # SCALE the uncertainty by the same factor used in PCA
        W_scaled = W / scaler.scale_[tracers.index(tracer)]
        W_tracers.append(W_scaled)
    
    W_tracers = np.array(W_tracers)

    # 4. Project W into PC space using PCA Loadings (Eigenvectors)
    # PC_uncert = sqrt(sum(loading_i^2 * W_scaled_i^2))
    loadings = pca.components_ # Shape (n_components, n_tracers)
    
    pc1_uncert = np.sqrt(np.sum((loadings[0, :]**2) * (W_tracers**2)))
    pc2_uncert = np.sqrt(np.sum((loadings[1, :]**2) * (W_tracers**2)))
    
    return pc1_uncert, pc2_uncert

def plot_event_pca_with_error(
    data, 
    site, 
    start_date, 
    end_date, 
    endmember_ids, 
    analytical_dict,  
    title="Event-Specific PCA"
):
    # Site-specific tracers
    if site == "Wade":
        #tracers = ['Ca_mg_L', 'Si_mg_L', 'Mg_mg_L', 'dD', 'd18O', 'Na_mg_L'] # Original Jan '26 WRR submission tracer selection
        tracers = ['Ca_mg_L', 'Na_mg_L', 'Mg_mg_L']                           # Aug '26 WRR resubmission traacer selection after more thourough vetting
    elif site == "Hungerford":
        #tracers = ['Ca_mg_L', 'Cl_mg_L', 'Si_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O'] # Original Jan '26 WRR submission tracer selection
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'Cu_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']  # Aug '26 WRR resubmission traacer selection after more thourough vetting  Ca, Cl, Cu, d18O, dD, K, Mg, Na
    elif site == "Potash":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    else:
        raise ValueError("Site not recognized. Use 'Wade', 'Potash', or 'Hungerford'.")
    
    # Ensure datetime formatting
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    # Subset Stream and Endmembers
    stream = data[(data["Site"] == site) & 
                  (data["Type"].isin(["Grab", "Grab/Isco", "Baseflow", "Isco"])) & 
                  (data["Date"] >= pd.to_datetime(start_date)) & 
                  (data["Date"] <= pd.to_datetime(end_date))].copy()

    endmembers = data[(data["Site"] == site) & 
                      (data["Sample ID"].isin(endmember_ids))].copy()

    subset_stream = stream[tracers].dropna().copy()
    subset_endmembers = endmembers[tracers].copy().fillna(endmembers[tracers].mean())
    subset_endmembers["Type"] = endmembers["Type"].values

    # --- PCA logic ---
    scaler = StandardScaler()
    scaled_stream = scaler.fit_transform(subset_stream[tracers])
    pca = PCA(n_components=2)
    stream_pca_result = pca.fit_transform(scaled_stream)
    
    scaled_endmembers = scaler.transform(subset_endmembers[tracers])
    endmember_pca_result = pca.transform(scaled_endmembers)
    subset_endmembers["PC1"] = endmember_pca_result[:, 0]
    subset_endmembers["PC2"] = endmember_pca_result[:, 1]

    # --- Compute PC Uncertainties ---
    em_types = subset_endmembers['Type'].unique()
    error_data = []

    for etype in em_types:
        etype_data = subset_endmembers[subset_endmembers['Type'] == etype]
        mean_pc1 = etype_data['PC1'].mean()
        mean_pc2 = etype_data['PC2'].mean()
        
        # Propagate error - FIX: use analytical_dict passed to function
        w_pc1, w_pc2 = calculate_pc_uncertainty(etype_data, tracers, scaler, pca, analytical_dict)
        
        error_data.append({
            'Type': etype, 'PC1_mean': mean_pc1, 'PC2_mean': mean_pc2,
            'PC1_err': w_pc1, 'PC2_err': w_pc2
        })
    
    stats_em = pd.DataFrame(error_data)

    # --- Plotting with propagated error bars ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Streamwater as background
    ax.scatter(stream_pca_result[:,0], stream_pca_result[:,1], marker='+', c='blue', alpha=0.3, label='Stream')

    # Endmembers with propagated error
    colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#2ca25f', '#636363', '#8856a7', '#d95f0e']
    markers = ['o', 's', '^', '*', '<', '>', 'D', 'P']

    for i, row in stats_em.iterrows():
        ax.errorbar(row['PC1_mean'], row['PC2_mean'], 
                    xerr=row['PC1_err'], yerr=row['PC2_err'],
                    fmt=markers[i % len(markers)], color=colors[i % len(colors)],
                    label=f"{row['Type']} (Propagated Error)", markersize=10, capsize=5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return stats_em, pca, scaler

##############################################
# function to explore end-memners in PCA space
##############################################

def plot_event_pca_explore_samples(
    data,
    site,
    start_date,
    end_date,
    exploratory_ids,
    title="Exploratory End-Member Projections"
):
    """
    Fits PCA on streamwater data for a specified site and date range, then 
    projects and plots individual exploratory candidate samples as distinct 
    points labeled by their Sample ID, color-coded by sample Type.
    Also labels streamwater samples with smaller Sample ID text.
    """
    # Site-specific tracers (August 2026 configuration)
    if site == "Wade":
        tracers = ['Ca_mg_L', 'Na_mg_L', 'Mg_mg_L']
    elif site == "Hungerford":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'Cu_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    elif site == "Potash":
        tracers = ['Ca_mg_L', 'Cl_mg_L', 'K_mg_L', 'Na_mg_L', 'Mg_mg_L', 'dD', 'd18O']
    else:
        raise ValueError("Site not recognized. Use 'Wade', 'Potash', or 'Hungerford'.")

    # Ensure datetime formatting
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y", errors="coerce")

    # 1. Subset streamwater (mixture) to fit the PCA space
    stream = data[
        (data["Site"] == site) &
        (data["Type"].isin(["Grab", "Grab/Isco", "Baseflow", "Isco"])) &
        (data["Date"] >= pd.to_datetime(start_date)) &
        (data["Date"] <= pd.to_datetime(end_date))
    ].copy()

    # 2. Subset exploratory candidate samples by Sample ID
    exploratory = data[
        (data["Site"] == site) &
        (data["Sample ID"].isin(exploratory_ids))
    ].copy()

    # Keep Sample ID alongside tracers for streamwater annotation
    subset_stream = stream[tracers + ['Sample ID']].dropna(subset=tracers).copy()
    
    # Handle missing values for exploratory samples (filling with mean if needed)
    subset_explore = exploratory[tracers].copy()
    subset_explore = subset_explore.fillna(subset_explore.mean())
    subset_explore["Sample ID"] = exploratory["Sample ID"].values
    subset_explore["Type"] = exploratory["Type"].values

    # 3. Fit PCA on streamwater data
    scaler = StandardScaler()
    scaled_stream = scaler.fit_transform(subset_stream[tracers])

    pca = PCA(n_components=2)
    stream_pca_result = pca.fit_transform(scaled_stream)
    subset_stream["PC1"] = stream_pca_result[:, 0]
    subset_stream["PC2"] = stream_pca_result[:, 1]

    # 4. Project exploratory samples into the stream PCA space
    scaled_explore = scaler.transform(subset_explore[tracers])
    explore_pca_result = pca.transform(scaled_explore)
    subset_explore["PC1"] = explore_pca_result[:, 0]
    subset_explore["PC2"] = explore_pca_result[:, 1]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    
    mpl.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    # Plot event streamwater background cloud (markers)
    ax.scatter(subset_stream["PC1"], subset_stream["PC2"], marker='o', s=110, c='none', edgecolors='royalblue', alpha=1, label='Streamwater')

    # Annotate each streamwater point with a smaller Sample ID label (~half the size of endmembers)
    for _, row in subset_stream.iterrows():
        ax.annotate(
            row["Sample ID"],
            (row["PC1"], row["PC2"]),
            xytext=(4, 4), textcoords='offset points',
            fontsize=5.5, fontweight='normal', color='royalblue', alpha=0.9
        )

    # Define color map for requested end-member types
    type_colors = {
        "Groundwater": "darkblue",
        "Baseflow": "deepskyblue",
        "Snowmelt lysimeter": "gold",
        "Snow": "yellow",
        "Soil water lysimeter": "red",
        "Precip": "green"
    }

    # Plot candidate end-members grouped by type for clean legend mapping
    for em_type, color in type_colors.items():
        type_subset = subset_explore[subset_explore["Type"] == em_type]
        if not type_subset.empty:
            ax.scatter(
                type_subset["PC1"], type_subset["PC2"],
                marker='o', s=130, c=color, edgecolors='black', zorder=5,
                label=em_type
            )

    # Annotate each candidate point with its Sample ID for identification
    for _, row in subset_explore.iterrows():
        # Match point label text color to its category color for readability
        txt_color = type_colors.get(row["Type"], "black")
        if txt_color == "yellow":  # Ensure high contrast for yellow text
            txt_color = "goldenrod"
            
        ax.annotate(
            row["Sample ID"],
            (row["PC1"], row["PC2"]),
            xytext=(6, 6), textcoords='offset points',
            fontsize=11, fontweight='bold', color=txt_color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8)
        )

    # Variance explained labels
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax.set_title(title)

    ax.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    plt.tight_layout()
    plt.show()

    return subset_explore, pca, scaler