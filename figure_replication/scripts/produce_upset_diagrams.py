#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import combinations
from upsetplot import from_contents, UpSet


# global style tweaks for publication‐quality
plt.rcParams.update({
    'figure.dpi': 600,
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'lines.linewidth': 2,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.format': 'pdf',
})

def compute_jaccard(sets_dict, output_dir, matrix_name):
    """
    Compute pairwise Jaccard similarity for the given sets_dict,
    save only long-format CSV for LaTeX heatmaps as jaccard_{matrix_name}.csv.
    """
    tools = list(sets_dict.keys())
    # Initialize DataFrame
    jaccard = pd.DataFrame(0.0, index=tools, columns=tools)
    # Compute similarities
    for a, b in combinations(tools, 2):
        A = sets_dict[a]
        B = sets_dict[b]
        union = A | B
        j_value = len(A & B) / len(union) if union else 0.0
        jaccard.at[a, b] = jaccard.at[b, a] = j_value
    # Fill diagonal
    for t in tools:
        jaccard.at[t, t] = 1.0

    # Convert to long format for LaTeX
    label_to_idx = {label: i for i, label in enumerate(jaccard.columns)}
    long_format = []
    for row in jaccard.index:
        for col in jaccard.columns:
            value = round(jaccard.at[row, col], 2)
            long_format.append({
                "x": label_to_idx[col],
                "y": label_to_idx[row],
                "c": value
            })
    long_df = pd.DataFrame(long_format)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save long-format CSV as the only CSV output
    path = os.path.join(output_dir, f"jaccard_{matrix_name}.csv")
    long_df.to_csv(path, index=False)

    # Print result
    print(f"Jaccard similarity matrix saved to {path} (long format)")
    return jaccard

def load_results(input_dir: str, results_name: str):
    """
    Scan each immediate subdirectory of input_dir for a CSV named
    {results_name}.csv, read it, and return a dict tool_name → DataFrame.
    """
    tool_dfs = {}
    for entry in sorted(os.listdir(input_dir)):
        if entry == 'mipi':
            continue
        sub = os.path.join(input_dir, entry)
        if os.path.isdir(sub):
            csv_path = os.path.join(sub, f"{results_name}.csv")
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                naming_convention = entry
                if entry == 'DL4PatchCorrectness':
                    naming_convention = 'Tian et al.'
                elif entry == 'LLM4PatchCorrectness':
                    naming_convention = 'LLM4PatchCorrect'
                elif entry == 'entropy_delta':
                    naming_convention = 'Yang et al.'
                assert {'patch_name','correctness','prediction'}.issubset(df.columns)
                tool_dfs[naming_convention] = df
    if not tool_dfs:
        raise ValueError(f"No '{results_name}.csv' files found under {input_dir}/*")
    return tool_dfs


def common_patch_names(tool_dfs):
    """Return the set of patch_names present in _every_ tool’s DataFrame."""
    sets = [set(df['patch_name']) for df in tool_dfs.values()]
    return set.intersection(*sets)


def extract_target_sets(tool_dfs, common, target_label):
    """
    For each tool, among the rows whose patch_name ∈ common,
    return the set of patch_name where correctness==prediction==target_label.
    """
    result = {}
    for tool, df in tool_dfs.items():
        dfc = df[df['patch_name'].isin(common)]
        sel = dfc[
            (dfc['correctness'] == target_label) &
            (dfc['prediction']  == target_label)
        ]
        result[tool] = set(sel['patch_name'])
    return result


def plot_upset(sets_dict, title, output_dir='.'):
    """
    Draw an UpSet plot optimized for research paper quality.
    """
    n = len(sets_dict)
    
    # Create figure with appropriate size and high resolution
    plt.figure(figsize=(10, 7))
    
    contents = from_contents(sets_dict)
    
    # Create UpSet plot with enhanced styling and smaller percentages
    upset = UpSet(contents, 
                 orientation='horizontal',
                 element_size=40,
                 show_percentages=True,
                 sort_by='cardinality',
                 min_subset_size=1
                 )
    
    # Apply plot with custom percentage formatting
    plot = upset.plot(fig=plt.gcf())
    
    # Reduce percentage text size if available in the plot components
    for ax in plt.gcf().get_axes():
        for text in ax.texts:
            if '%' in text.get_text():  # Identify percentage labels
                text.set_fontsize(14)    # Set smaller font size for percentages
                new_text = text.get_text().replace('%', '')
                text.set_text(new_text)
    
    # Enhance title and overall appearance
    plt.title(title, x=0.5, y=0.98, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontweight='bold', fontsize = 22)

    # Find the side plot axis and set the title
    side_ax = plt.gcf().get_axes()[0]  # Access the side plot axes
    side_ax.set_title('Recall (%)', x=0, fontsize=18)
    side_ax = plt.gcf().get_axes()[3]  
    #side_ax.set_ylabel('Intersection size (%)', fontsize=18)
     

    plt.tight_layout()
    
    # Save figure in high quality
    # sanitize filename
    fn = title.lower().replace(' ', '_').replace(':', '')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to the specified directory
    output_path = os.path.join(output_dir, f"{fn}.png")
    plt.savefig(output_path, dpi=600)
    print(f"→ saved {output_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Plot UpSet diagrams of correct vs overfitting patches")
    p.add_argument("results_name", help="basename (no .csv) of each results file")
    p.add_argument("--input_dir", required=True, help="parent directory containing one subfolder per tool")
    p.add_argument("--output_dir", "-o", default=".", help="directory to save output diagrams (default: current directory)")
    args = p.parse_args()

    # 1) load all tools
    tool_dfs = load_results(args.input_dir, args.results_name)

    # 2) find patches common to _all_ tools
    common = common_patch_names(tool_dfs)
    print(f"Found {len(common)} patches common across {len(tool_dfs)} tools")

    # 3) build sets for both categories
    correct_sets    = extract_target_sets(tool_dfs, common, 'correct')
    overfit_sets    = extract_target_sets(tool_dfs, common, 'overfitting')

    # 4) plot with output directory
    plot_upset(correct_sets,
              title="Correctly Classified: Correct Patches",
              output_dir=args.output_dir)
    plot_upset(overfit_sets,
              title="Correctly Classified: Overfitting Patches",
              output_dir=args.output_dir)
    
    # 5) compute Jaccard matrices
    compute_jaccard(correct_sets, args.output_dir, 'correct')
    compute_jaccard(overfit_sets, args.output_dir, 'overfitting')


if __name__ == "__main__":
    main()
