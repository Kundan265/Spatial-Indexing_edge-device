"""
Plotting module: generates all figures from benchmark results CSV.
Log-log scaling plots, sensitivity curves, grouped bar charts,
and theoretical vs. empirical overlays.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Style constants
COLORS = {'kdtree': '#1f77b4', 'octree': '#ff7f0e', 'svo': '#2ca02c'}
MARKERS = {'kdtree': 'o', 'octree': 's', 'svo': '^'}
LABELS = {'kdtree': 'k-d Tree', 'octree': 'Octree', 'svo': 'SVO'}
FIG_DPI = 300
FIG_SIZE = (8, 5)


def _setup_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.figsize': FIG_SIZE,
    })


def load_results(csv_path):
    """Load benchmark results CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    numeric_cols = [
        'build_time_s', 'knn_mean_us', 'knn_median_us', 'knn_std_us',
        'knn_p99_us', 'radius_mean_us', 'radius_median_us', 'radius_std_us',
        'radius_p99_us', 'peak_memory_bytes', 'struct_memory_bytes',
        'node_count', 'leaf_count', 'internal_count', 'max_depth',
        'compression_ratio', 'avg_radius_result_size',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _has_structural_cols(df):
    """Check if DataFrame has the structural stats columns."""
    return all(c in df.columns for c in ['leaf_count', 'internal_count', 'max_depth'])


# =============================================================================
# Round 1: Scalability plots (log-log)
# =============================================================================

def plot_round1_build_time(df, output_dir):
    """Log-log: Build time vs N."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N')['build_time_s'].agg(['median', 'std']).reset_index()
        ax.errorbar(grouped['N'], grouped['median'], yerr=grouped['std'],
                    marker=MARKERS[sname], color=COLORS[sname],
                    label=LABELS[sname], capsize=3, linewidth=2)

    # Theoretical O(N log N) overlay
    ns = r1['N'].unique()
    ns = np.sort(ns)
    if len(ns) > 0:
        ref = r1[(r1['structure'] == 'kdtree') & (r1['N'] == ns[0])]['build_time_s'].median()
        theory = ref * (ns * np.log2(ns)) / (ns[0] * np.log2(ns[0]))
        ax.plot(ns, theory, '--', color='gray', alpha=0.5, label='O(N log N) ref')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('Build Time (seconds)')
    ax.set_title('Build Time vs Dataset Size')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round1_build_time.png'), dpi=FIG_DPI)
    plt.close(fig)


def plot_round1_knn_latency(df, output_dir):
    """Log-log: KNN query latency vs N."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N')['knn_median_us'].agg(['median', 'std']).reset_index()
        ax.errorbar(grouped['N'], grouped['median'], yerr=grouped['std'],
                    marker=MARKERS[sname], color=COLORS[sname],
                    label=LABELS[sname], capsize=3, linewidth=2)

    # Theoretical O(log N) overlay
    ns = np.sort(r1['N'].unique())
    if len(ns) > 0:
        ref = r1[(r1['structure'] == 'kdtree') & (r1['N'] == ns[0])]['knn_median_us'].median()
        theory = ref * np.log2(ns) / np.log2(ns[0])
        ax.plot(ns, theory, '--', color='gray', alpha=0.5, label='O(log N) ref')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('KNN Query Latency (us)')
    ax.set_title('KNN Query Latency vs Dataset Size (k=10)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round1_knn_latency.png'), dpi=FIG_DPI)
    plt.close(fig)


def plot_round1_radius_latency(df, output_dir):
    """Log-log: Radius query latency vs N."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N')['radius_median_us'].agg(['median', 'std']).reset_index()
        ax.errorbar(grouped['N'], grouped['median'], yerr=grouped['std'],
                    marker=MARKERS[sname], color=COLORS[sname],
                    label=LABELS[sname], capsize=3, linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('Radius Query Latency (us)')
    ax.set_title('Radius Query Latency vs Dataset Size (r=0.5)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round1_radius_latency.png'), dpi=FIG_DPI)
    plt.close(fig)


def plot_round1_memory(df, output_dir):
    """Log-log: Peak memory vs N."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N')['peak_memory_bytes'].agg(['median', 'std']).reset_index()
        grouped['median_mb'] = grouped['median'] / (1024 * 1024)
        grouped['std_mb'] = grouped['std'] / (1024 * 1024)
        ax.errorbar(grouped['N'], grouped['median_mb'], yerr=grouped['std_mb'],
                    marker=MARKERS[sname], color=COLORS[sname],
                    label=LABELS[sname], capsize=3, linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Peak Memory Usage vs Dataset Size')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round1_memory.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Round 2A: KNN sensitivity
# =============================================================================

def plot_round2a_knn_sensitivity(df, output_dir):
    """KNN latency vs k."""
    _setup_style()
    r2a = df[df['round'] == 'round2a']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r2a[r2a['structure'] == sname]
        grouped = sub.groupby('k')['knn_median_us'].agg(['median', 'std']).reset_index()
        ax.errorbar(grouped['k'], grouped['median'], yerr=grouped['std'],
                    marker=MARKERS[sname], color=COLORS[sname],
                    label=LABELS[sname], capsize=3, linewidth=2)

    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('KNN Query Latency (us)')
    ax.set_title('KNN Sensitivity: Latency vs k (N=1M, uniform)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round2a_knn_sensitivity.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Round 2B: Radius sensitivity (dual-axis)
# =============================================================================

def plot_round2b_radius_sensitivity(df, output_dir):
    """Dual-axis: Radius latency + result set size vs r."""
    _setup_style()
    r2b = df[df['round'] == 'round2b']
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    ax2 = ax1.twinx()

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r2b[r2b['structure'] == sname]
        grouped = sub.groupby('radius').agg({
            'radius_median_us': ['median', 'std'],
            'avg_radius_result_size': 'median'
        }).reset_index()
        grouped.columns = ['radius', 'lat_median', 'lat_std', 'result_size']

        ax1.errorbar(grouped['radius'], grouped['lat_median'], yerr=grouped['lat_std'],
                     marker=MARKERS[sname], color=COLORS[sname],
                     label=f"{LABELS[sname]} latency", capsize=3, linewidth=2)
        ax2.plot(grouped['radius'], grouped['result_size'],
                 '--', marker=MARKERS[sname], color=COLORS[sname],
                 alpha=0.5, linewidth=1)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Radius Query Latency (us)')
    ax2.set_ylabel('Avg Result Set Size (dashed)')
    ax1.set_title('Radius Sensitivity: Latency & Result Size vs r (N=1M, uniform)')
    ax1.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round2b_radius_sensitivity.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Round 3: Distribution sensitivity (grouped bar charts)
# =============================================================================

def _grouped_bar(ax, df_round3, metric, ylabel, title, structures, distributions):
    """Helper for grouped bar charts."""
    n_dists = len(distributions)
    n_structs = len(structures)
    x = np.arange(n_dists)
    width = 0.25

    for i, sname in enumerate(structures):
        vals = []
        errs = []
        for dist in distributions:
            sub = df_round3[(df_round3['structure'] == sname) &
                            (df_round3['distribution'] == dist)]
            vals.append(sub[metric].median())
            errs.append(sub[metric].std())
        ax.bar(x + i * width, vals, width, yerr=errs,
               label=LABELS[sname], color=COLORS[sname],
               capsize=3, alpha=0.85)

    ax.set_xlabel('Distribution')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.title() for d in distributions], rotation=15)
    ax.legend()


def plot_round3_distribution(df, output_dir):
    """4 grouped bar charts for Round 3."""
    _setup_style()
    r3 = df[df['round'] == 'round3']
    structures = ['kdtree', 'octree', 'svo']
    distributions = r3['distribution'].unique().tolist()

    metrics = [
        ('build_time_s', 'Build Time (s)', 'Build Time by Distribution'),
        ('knn_median_us', 'KNN Latency (us)', 'KNN Latency by Distribution'),
        ('radius_median_us', 'Radius Latency (us)', 'Radius Latency by Distribution'),
        ('peak_memory_bytes', 'Peak Memory (bytes)', 'Memory Usage by Distribution'),
    ]

    for metric, ylabel, title in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        _grouped_bar(ax, r3, metric, ylabel, title, structures, distributions)
        fig.tight_layout()
        fname = f"round3_{metric.replace('_us', '').replace('_s', '').replace('_bytes', '')}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=FIG_DPI)
        plt.close(fig)


# =============================================================================
# Bonus: SVO compression ratio
# =============================================================================

def plot_compression_ratio(df, output_dir):
    """SVO compression ratio by N and distribution."""
    _setup_style()

    # By N (Round 1)
    r1_svo = df[(df['round'] == 'round1') & (df['structure'] == 'svo')]
    if not r1_svo.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        grouped = r1_svo.groupby('N')['compression_ratio'].median().reset_index()
        ax.plot(grouped['N'], grouped['compression_ratio'], 'g^-', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Dataset Size N')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('SVO Compression Ratio vs N (uniform)')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'svo_compression_vs_n.png'), dpi=FIG_DPI)
        plt.close(fig)

    # By distribution (Round 3)
    r3_svo = df[(df['round'] == 'round3') & (df['structure'] == 'svo')]
    if not r3_svo.empty:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        grouped = r3_svo.groupby('distribution')['compression_ratio'].median()
        ax.bar(range(len(grouped)), grouped.values, color='#2ca02c', alpha=0.85)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels([d.title() for d in grouped.index], rotation=15)
        ax.set_ylabel('Compression Ratio')
        ax.set_title('SVO Compression Ratio by Distribution (N=1M)')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'svo_compression_by_dist.png'), dpi=FIG_DPI)
        plt.close(fig)


# =============================================================================
# Theoretical vs empirical overlay
# =============================================================================

def plot_theory_vs_empirical(df, output_dir):
    """Overlay theoretical curves on empirical data for each structure."""
    _setup_style()
    r1 = df[df['round'] == 'round1']

    for metric, theory_fn, ylabel, title_suffix in [
        ('build_time_s', lambda n: n * np.log2(n), 'Build Time (s)', 'Build Time'),
        ('knn_median_us', lambda n: np.log2(n), 'KNN Latency (us)', 'KNN Latency'),
    ]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for sname in ['kdtree', 'octree', 'svo']:
            sub = r1[r1['structure'] == sname]
            grouped = sub.groupby('N')[metric].median().reset_index()
            ns = grouped['N'].values
            vals = grouped[metric].values

            ax.plot(ns, vals, marker=MARKERS[sname], color=COLORS[sname],
                    label=f"{LABELS[sname]} (measured)", linewidth=2)

            # Theoretical overlay normalized to first point
            if len(ns) > 0 and vals[0] > 0:
                theory_vals = np.array([theory_fn(n) for n in ns])
                scale = vals[0] / theory_vals[0]
                ax.plot(ns, scale * theory_vals, '--', color=COLORS[sname],
                        alpha=0.4, label=f"{LABELS[sname]} (theoretical)")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Dataset Size N')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Theory vs Empirical: {title_suffix}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = f"theory_vs_empirical_{metric.split('_')[0]}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=FIG_DPI)
        plt.close(fig)


# =============================================================================
# Enhanced: P99 tail latency comparison
# =============================================================================

def plot_round1_p99_latency(df, output_dir):
    """Log-log: P99 KNN and Radius latency vs N (shows worst-case behavior)."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N').agg({
            'knn_p99_us': 'median',
            'radius_p99_us': 'median'
        }).reset_index()
        ax1.plot(grouped['N'], grouped['knn_p99_us'],
                 marker=MARKERS[sname], color=COLORS[sname],
                 label=LABELS[sname], linewidth=2)
        ax2.plot(grouped['N'], grouped['radius_p99_us'],
                 marker=MARKERS[sname], color=COLORS[sname],
                 label=LABELS[sname], linewidth=2)

    for ax, title in [(ax1, 'P99 KNN Latency vs N'), (ax2, 'P99 Radius Latency vs N')]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Dataset Size N')
        ax.set_ylabel('P99 Latency (us)')
        ax.set_title(title)
        ax.legend()

    fig.suptitle('Tail Latency (99th Percentile)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round1_p99_latency.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Enhanced: Combined scalability overview (2x2)
# =============================================================================

def plot_round1_overview(df, output_dir):
    """2x2 overview: build time, KNN latency, radius latency, memory vs N."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('build_time_s', 'Build Time (s)', axes[0, 0]),
        ('knn_median_us', 'KNN Latency (us)', axes[0, 1]),
        ('radius_median_us', 'Radius Latency (us)', axes[1, 0]),
        ('peak_memory_bytes', 'Peak Memory (MB)', axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        for sname in ['kdtree', 'octree', 'svo']:
            sub = r1[r1['structure'] == sname]
            grouped = sub.groupby('N')[metric].agg(['median', 'std']).reset_index()
            y = grouped['median']
            if 'memory' in metric:
                y = y / (1024 * 1024)
                ylabel = 'Peak Memory (MB)'
            ax.errorbar(grouped['N'], y, yerr=grouped['std'] / (1024*1024) if 'memory' in metric else grouped['std'],
                        marker=MARKERS[sname], color=COLORS[sname],
                        label=LABELS[sname], capsize=3, linewidth=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    fig.suptitle('Scalability Overview (k=10, r=0.5, uniform)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'round1_overview.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Enhanced: Latency heatmap
# =============================================================================

def plot_latency_heatmap(df, output_dir):
    """Heatmap: KNN median latency across structures and distributions."""
    _setup_style()
    r3 = df[df['round'] == 'round3']
    if r3.empty:
        return

    structures = ['kdtree', 'octree', 'svo']
    distributions = r3['distribution'].unique().tolist()

    # KNN heatmap
    data = np.zeros((len(structures), len(distributions)))
    for i, s in enumerate(structures):
        for j, d in enumerate(distributions):
            sub = r3[(r3['structure'] == s) & (r3['distribution'] == d)]
            data[i, j] = sub['knn_median_us'].median()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(distributions)))
    ax.set_xticklabels([d.title() for d in distributions], rotation=15)
    ax.set_yticks(range(len(structures)))
    ax.set_yticklabels([LABELS[s] for s in structures])

    # Annotate cells
    for i in range(len(structures)):
        for j in range(len(distributions)):
            text_color = 'white' if data[i, j] > data.max() * 0.6 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                    color=text_color, fontweight='bold')

    ax.set_title('KNN Median Latency (us) by Structure x Distribution')
    fig.colorbar(im, ax=ax, label='Latency (us)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'latency_heatmap.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Enhanced: Build efficiency (build time per million points)
# =============================================================================

def plot_build_efficiency(df, output_dir):
    """Build time per million points vs N — shows amortized construction cost."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N')['build_time_s'].median().reset_index()
        grouped['per_million'] = grouped['build_time_s'] / (grouped['N'] / 1e6)
        ax.plot(grouped['N'], grouped['per_million'],
                marker=MARKERS[sname], color=COLORS[sname],
                label=LABELS[sname], linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('Build Time per Million Points (s)')
    ax.set_title('Build Efficiency: Time per Million Points vs N')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'build_efficiency.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Structural stats plots (require new CSV columns)
# =============================================================================

def plot_structural_node_breakdown(df, output_dir):
    """Stacked bar: leaf vs internal nodes by N."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    structures = ['kdtree', 'octree', 'svo']
    ns = sorted(r1['N'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, sname in zip(axes, structures):
        leaves = []
        internals = []
        for N in ns:
            sub = r1[(r1['structure'] == sname) & (r1['N'] == N)]
            leaves.append(sub['leaf_count'].median())
            internals.append(sub['internal_count'].median())

        x = range(len(ns))
        ax.bar(x, internals, label='Internal', color=COLORS[sname], alpha=0.9)
        ax.bar(x, leaves, bottom=internals, label='Leaf', color=COLORS[sname], alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{n//1000}K' if n < 1e6 else f'{n//1000000}M' for n in ns],
                           rotation=45, fontsize=8)
        ax.set_title(LABELS[sname])
        ax.set_ylabel('Node Count')
        ax.set_yscale('log')
        ax.legend(fontsize=8)

    fig.suptitle('Node Breakdown: Leaf vs Internal Nodes by N', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'structural_node_breakdown.png'), dpi=FIG_DPI)
    plt.close(fig)


def plot_structural_depth_comparison(df, output_dir):
    """Tree depth vs N for all structures."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N')['max_depth'].median().reset_index()
        ax.plot(grouped['N'], grouped['max_depth'],
                marker=MARKERS[sname], color=COLORS[sname],
                label=LABELS[sname], linewidth=2)

    # Theoretical curves
    ns = np.sort(r1['N'].unique()).astype(float)
    if len(ns) > 0:
        ax.plot(ns, np.log2(ns / 32), '--', color='gray', alpha=0.4, label='log2(N/32) ref')
        ax.plot(ns, np.log(ns / 32) / np.log(8), ':', color='gray', alpha=0.4,
                label='log8(N/32) ref')

    ax.set_xscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('Max Tree Depth')
    ax.set_title('Tree Depth vs Dataset Size')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'structural_depth.png'), dpi=FIG_DPI)
    plt.close(fig)


def plot_structural_memory_breakdown(df, output_dir):
    """Compare peak_memory vs struct_memory for all structures."""
    _setup_style()
    r1 = df[df['round'] == 'round1']
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sname in ['kdtree', 'octree', 'svo']:
        sub = r1[r1['structure'] == sname]
        grouped = sub.groupby('N').agg({
            'peak_memory_bytes': 'median',
            'struct_memory_bytes': 'median'
        }).reset_index()

        ax.plot(grouped['N'], grouped['struct_memory_bytes'] / 1e6,
                marker=MARKERS[sname], color=COLORS[sname],
                label=f'{LABELS[sname]} (struct)', linewidth=2)
        ax.plot(grouped['N'], grouped['peak_memory_bytes'] / 1e6,
                '--', marker=MARKERS[sname], color=COLORS[sname],
                alpha=0.4, label=f'{LABELS[sname]} (peak)', linewidth=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dataset Size N')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory: Structure Size vs Peak Allocation')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'structural_memory.png'), dpi=FIG_DPI)
    plt.close(fig)


# =============================================================================
# Master function
# =============================================================================

def generate_all_plots(csv_path, output_dir):
    """Generate all plots from benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(csv_path)

    print("Generating plots...")
    if 'round1' in df['round'].values:
        print("  Round 1: scaling plots...")
        plot_round1_build_time(df, output_dir)
        plot_round1_knn_latency(df, output_dir)
        plot_round1_radius_latency(df, output_dir)
        plot_round1_memory(df, output_dir)
        print("  Round 1: enhanced plots...")
        plot_round1_p99_latency(df, output_dir)
        plot_round1_overview(df, output_dir)
        plot_build_efficiency(df, output_dir)

    if 'round2a' in df['round'].values:
        print("  Round 2A: KNN sensitivity...")
        plot_round2a_knn_sensitivity(df, output_dir)

    if 'round2b' in df['round'].values:
        print("  Round 2B: Radius sensitivity...")
        plot_round2b_radius_sensitivity(df, output_dir)

    if 'round3' in df['round'].values:
        print("  Round 3: Distribution sensitivity...")
        plot_round3_distribution(df, output_dir)
        print("  Latency heatmap...")
        plot_latency_heatmap(df, output_dir)

    print("  Compression ratio...")
    plot_compression_ratio(df, output_dir)

    print("  Theory vs empirical...")
    plot_theory_vs_empirical(df, output_dir)

    if _has_structural_cols(df) and 'round1' in df['round'].values:
        print("  Structural stats...")
        plot_structural_node_breakdown(df, output_dir)
        plot_structural_depth_comparison(df, output_dir)
        plot_structural_memory_breakdown(df, output_dir)

    print(f"All plots saved to {output_dir}/")


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base, 'results', 'benchmark_results.csv')
    output_dir = os.path.join(base, 'plots')
    generate_all_plots(csv_path, output_dir)
