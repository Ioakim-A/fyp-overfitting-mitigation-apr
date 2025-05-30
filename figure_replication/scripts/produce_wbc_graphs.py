import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

name_conversion = {
    'entropy_delta':        'Yang et al.',
    'Invalidator':          'Invalidator',
    'FIXCHECK':             'FIXCHECK',
    'LLM4PatchCorrectness': 'LLM4PatchCorrect',
    'DL4PatchCorrectness':  'Tian et al.',
}

def parse_ci(s):
    """
    Parse strings like:
      "0.64 (0.56-0.72)"   → (0.64, 0.56, 0.72)
      "-0.02 (-0.11-0.08)" → (-0.02, -0.11, 0.08)
    """
    pattern = re.compile(
        r'^\s*([+-]?\d+(?:\.\d+)?)\s*'       # mean
        r'\(\s*([+-]?\d+(?:\.\d+)?)\s*-\s*'  # lower
        r'([+-]?\d+(?:\.\d+)?)\s*\)\s*$'     # upper
    )
    m = pattern.match(s)
    if not m:
        raise ValueError(f"Could not parse CI from '{s}'")
    return map(float, m.groups())

def calculate_wpc_metrics(p, num_overfitting, num_correct):
    TP = num_correct*(1-p)
    FP = num_overfitting*p
    TN = num_overfitting*p
    FN = num_correct*(1-p)
    accuracy = (TP+TN)/(num_correct+num_overfitting)
    pos_rec = TP/num_correct        if num_correct    else 0
    neg_rec = TN/num_overfitting    if num_overfitting else 0
    bal_acc = (pos_rec+neg_rec)/2
    prec    = TP/(TP+FP)            if (TP+FP)        else 0
    f1      = 2*prec*pos_rec/(prec+pos_rec) if (prec+pos_rec) else 0
    num = TP*TN - FP*FN
    den = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc = num/den if den else 0
    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": bal_acc,
        "Precision": prec,
        "Positive Recall": pos_rec,
        "Negative Recall": neg_rec,
        "F1 Score": f1,
        "MCC": mcc
    }

def main():
    parser = argparse.ArgumentParser(
        description='Plot and summarize 6 metrics for tools vs. WPC with shaded WPC area'
    )
    parser.add_argument('csv_file',
                        help='CSV: Tool + 7 metric columns as "mean (lo-hi)"')
    parser.add_argument('num_correct',    type=int, help='Count of positive-class examples')
    parser.add_argument('num_overfitting', type=int, help='Count of negative-class examples')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    metrics = ["Accuracy", "Precision", "Positive Recall",
               "Negative Recall", "F1 Score", "MCC"]
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Missing '{m}' column in CSV")

    # Precompute WPC curves
    prob  = np.linspace(0.5, 1.0, 500)
    x_wpc = prob * 100
    wpc   = {
        m: np.array([
            calculate_wpc_metrics(p, args.num_overfitting, args.num_correct)[m]
            for p in prob
        ])
        for m in metrics
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(14,9), dpi=300)
    cmap = plt.get_cmap('tab10')

    for idx, metric in enumerate(metrics):
        ax  = axes.flat[idx]
        y_w = wpc[metric]

        # --- compute y-axis clamp first ---
        ci_vals = []
        for _, row in df.iterrows():
            _, lo, hi = parse_ci(row[metric])
            ci_vals.extend([lo, hi])
        all_lo_hi = ci_vals + list(y_w)
        dmin, dmax = min(all_lo_hi), max(all_lo_hi)
        pad = (dmax - dmin) * 0.05
        if metric == "MCC":
            y_low = max(-1, dmin - pad)
            y_high = min( 1, dmax + pad)
        else:
            y_low = max( 0, dmin - pad)
            y_high = min( 1, dmax + pad)

        # Shade under the WPC curve down to y_low
        ax.fill_between(
            x_wpc,
            y_w,
            y_low,
            color='grey',
            alpha=0.2,
            zorder=1
        )
        # Plot the WPC curve on top
        ax.plot(
            x_wpc, y_w,
            'k-', lw=2,
            zorder=2,
            label='WPC' if idx==0 else None
        )

        # sort for interpolation
        si       = np.argsort(y_w)
        y_sorted = y_w[si]
        x_sorted = x_wpc[si]
        ymin, ymax = y_w.min(), y_w.max()

        # bucket tools
        intersect, above, below = [], [], []
        for ti, row in df.iterrows():
            mean, lo, hi = parse_ci(row[metric])
            color = cmap(ti % 10)
            data  = dict(
                tool  = name_conversion[row['Tool']],
                mean  = mean,
                lo    = lo,
                hi    = hi,
                color = color
            )
            if ymin <= lo <= ymax:
                data['x'] = float(np.interp(lo, y_sorted, x_sorted))
                intersect.append(data)
            elif lo > ymax:
                above.append(data)
            else:
                below.append(data)

        # --- PRINT SUMMARY FOR THIS METRIC ---
        print(f"\n=== {metric} ===")
        print(f"WPC range: {ymin:.3f} – {ymax:.3f}")
        if above:
            print("Tools always above WPC (lower CI > max WPC):")
            for d in sorted(above, key=lambda d:(d['lo'], d['mean'])):
                print(f"  {d['tool']}: mean={d['mean']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}]")
        else:
            print("No tool always above WPC.")
        if intersect:
            print("Tools intersecting WPC:")
            for d in sorted(intersect, key=lambda d:(d['x'], d['lo'])):
                print(f"  {d['tool']}: x_int={d['x']:.1f}%, mean={d['mean']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}]")
        else:
            print("No tool intersects WPC.")
        if below:
            print("Tools always below WPC (lower CI < min WPC):")
            for d in sorted(below, key=lambda d:(d['lo'], d['mean'])):
                print(f"  {d['tool']}: mean={d['mean']:.3f}, CI=[{d['lo']:.3f}, {d['hi']:.3f}]")
        else:
            print("No tool always below WPC.")
        # --------------------------------------

        # plotting points / errorbars
        if not intersect:
            all_tools = above + below
            all_tools.sort(key=lambda d:(d['lo'], d['mean']))
            spots = np.linspace(50, 100, len(all_tools)+2)[1:-1]
            for d, xpos in zip(all_tools, spots):
                ax.errorbar(
                    xpos, d['mean'],
                    yerr=[[d['mean']-d['lo']], [d['hi']-d['mean']]],
                    fmt='s', color=d['color'], ecolor=d['color'],
                    elinewidth=2, capsize=5,
                    label=d['tool'] if idx==0 else None
                )
        else:
            for d in intersect:
                ax.errorbar(
                    d['x'], d['mean'],
                    yerr=[[d['mean']-d['lo']], [d['hi']-d['mean']]],
                    fmt='s', color=d['color'], ecolor=d['color'],
                    elinewidth=2, capsize=5,
                    label=d['tool'] if idx==0 else None
                )
            xs = [d['x'] for d in intersect]
            x_min_int, x_max_int = min(xs), max(xs)
            if above:
                above.sort(key=lambda d:(d['lo'], d['mean']))
                spots = np.linspace(x_max_int, 100, len(above)+2)[1:-1]
                for d, xpos in zip(above, spots):
                    ax.errorbar(
                        xpos, d['mean'],
                        yerr=[[d['mean']-d['lo']], [d['hi']-d['mean']]],
                        fmt='s', color=d['color'], ecolor=d['color'],
                        elinewidth=2, capsize=5,
                        label=d['tool'] if idx==0 else None
                    )
            if below:
                below.sort(key=lambda d:(d['lo'], d['mean']))
                spots = np.linspace(50, x_min_int, len(below)+2)[1:-1]
                for d, xpos in zip(below, spots):
                    ax.errorbar(
                        xpos, d['mean'],
                        yerr=[[d['mean']-d['lo']], [d['hi']-d['mean']]],
                        fmt='s', color=d['color'], ecolor=d['color'],
                        elinewidth=2, capsize=5,
                        label=d['tool'] if idx==0 else None
                    )

        # formatting subplot
        ax.set_title(metric, fontsize=20, weight='bold')
        ax.set_xlim(50, 100)
        ax.set_xlabel('WPC Guess “Overfitting” (%)', fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_ylim(y_low, y_high)

    # single legend below
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(labels),
               fontsize=18,
               frameon=True,
               edgecolor='gray',
               bbox_to_anchor=(0.5, 0.025))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.3, wspace=0.25)

    out = args.out
    output_path = Path(args.out)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, bbox_inches='tight')
    print(f"\nSaved figure as {out}")

if __name__ == "__main__":
    main()
