import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

# Dictionary to convert tool names in CSV to display names
name_conversion = {
    'entropy_delta': 'Yang et al.',
    'Invalidator': 'Invalidator',
    'FIXCHECK': 'FIXCHECK',
    'LLM4PatchCorrectness': 'LLM4Patch\nCorrect',
    'DL4PatchCorrectness': 'Tian et al.',
}

baseline_tools = ['WBC', 'RS']


def parse_value_and_ci(val_str):
    """
    Parse a string that may contain a confidence interval of format 'val (low-high)'.
    Supports negative numbers. Returns (value, lower_err, upper_err).
    If no CI present or parse fails, returns (value_or_0, 0.0, 0.0).
    """
    # Regex for numbers with optional sign and decimals
    pattern = r"^\s*(-?[0-9]*\.?[0-9]+)\s*\(\s*(-?[0-9]*\.?[0-9]+)\s*-\s*(-?[0-9]*\.?[0-9]+)\s*\)\s*$"
    match = re.match(pattern, val_str)
    if match:
        val = float(match.group(1))
        lo = float(match.group(2))
        hi = float(match.group(3))
        return val, val - lo, hi - val
    # Fallback: try to parse a standalone number
    try:
        val = float(val_str)
        return val, 0.0, 0.0
    except ValueError:
        return 0.0, 0.0, 0.0


def create_bar_chart(csv_file, output_dir='.', use_ci=False):
    """
    Create a bar chart comparing metrics across different tools.
    If use_ci is True, expects values formatted as 'val (low-high)' for non-baseline tools and plots error bars.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    tools = df['Tool'].tolist()
    metrics = df.columns[1:].tolist()

    # reorder tools so baselines are last
    regular_tools = [t for t in tools if t not in baseline_tools]
    tools = regular_tools + [t for t in baseline_tools if t in tools]

    # Plot styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 23,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22
    })

    fig, ax = plt.subplots(figsize=(18, 10))
    bar_width = 0.8 / len(tools) 
    positions = np.arange(len(metrics))
    colors = plt.cm.viridis(np.linspace(0, 1, len(tools)))

    for i, tool in enumerate(tools):
        is_baseline = tool in baseline_tools
        raw_vals = df[df['Tool'] == tool].iloc[0]

        values = []
        lower_errs = []
        upper_errs = []

        for metric in metrics:
            raw = raw_vals[metric]
            if pd.isna(raw) or str(raw) == 'N/A':
                val, lo, hi = 0.0, 0.0, 0.0
            else:
                val, lo, hi = parse_value_and_ci(str(raw))

            values.append(val)
            if use_ci and not is_baseline:
                lower_errs.append(lo)
                upper_errs.append(hi)

        # X positions for bars
        xpos = positions + (i - len(tools)/2 + 0.5) * bar_width
        display_name = name_conversion.get(tool, tool)

        # Determine errorbars
        if use_ci and not is_baseline:
            yerr = np.vstack([lower_errs, upper_errs])
            capsize = 2.8
        else:
            yerr = None
            capsize = 0

        # Plot bars
        ax.bar(
            xpos,
            values,
            width=bar_width,
            color=colors[i],
            label=display_name,
            alpha=0.5 if is_baseline else 0.8,
            edgecolor='black',
            linewidth=1,
            hatch='//' if is_baseline else None,
            yerr=yerr,
            capsize=capsize,
        )

    # Labels, title, grid
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xticks(positions)
    metrics[1] = 'Balanced\nAccuracy'
    metrics[3] = 'Positive\nRecall'
    metrics[4] = 'Negative\nRecall'
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Legend
    if not 'repairllama' in output_dir:
      ax.legend(
          loc='upper center',
          bbox_to_anchor=(0.5, -0.1),
          ncol=len(tools),
          fancybox=True,
          shadow=True,
          fontsize=20
      )

    dname = 'classical'
    if 'repairllama' in output_dir: dname = 'repairllama'
    plt.text(0.98, 0.98, dname+' dataset', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontweight='bold', fontsize=24)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_bar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create bar charts from CSV data.')
    parser.add_argument('--input_csv', required=True, help='Path to the CSV file')
    parser.add_argument('--output_dir', default='./plots', help='Directory to save the plot')
    parser.add_argument('--with-ci', action='store_true', help='Parse and plot confidence intervals for non-baseline tools')
    args = parser.parse_args()

    create_bar_chart(args.input_csv, args.output_dir, use_ci=args.with_ci)
