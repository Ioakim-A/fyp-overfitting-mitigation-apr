import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import os
from scipy.stats import linregress


def extract_ratio(bug_string):
    """Extract the correct:overfitting ratio from the bug label."""
    match = re.search(r'\((\d+):(\d+)\)', bug_string)
    if match:
        correct = int(match.group(1))
        overfitting = int(match.group(2))
        if overfitting == 0:
            return float('inf') if correct > 0 else 0
        return correct / overfitting
    return None


def process_and_plot(csv_path, num_deciles=10, output_file=None):
    """Process the CSV data, perform linear regression on decile F1 vs difficulty, and plot the results."""
    # Read the CSV file except the last row
    data = pd.read_csv(csv_path)
    data = data.iloc[:-1]
    
    # Extract ratio and add as new column
    data['ratio'] = data['Bug'].apply(extract_ratio)
    
    # Extract correct counts
    correct_counts = data['Bug'].str.extract(r'\((\d+):', expand=False).astype(int)
    
    # Filter
    data = data.dropna(subset=['ratio'])
    data = data[correct_counts > 0]
    print(len(data), "bugs with correct patches found.")
    if data.empty:
        print("No bugs with correct patches found.")
        return

    # Sort by ratio descending
    data = data.sort_values(by='ratio', ascending=False)

    # Adjust deciles if too few points
    total_bugs = len(data)
    if total_bugs < num_deciles:
        print(f"Warning: Too few data points ({total_bugs}) for {num_deciles} deciles.")
        num_deciles = total_bugs

    # Divide into deciles
    base_size = total_bugs // num_deciles
    remainder = total_bugs % num_deciles
    deciles = []
    idx = 0
    for i in range(num_deciles):
        size = base_size + (1 if i < remainder else 0)
        deciles.append(data.iloc[idx:idx+size])
        idx += size

    # Compute average F1 per tool per decile
    tool_cols = [c for c in data.columns if c not in ['Bug', 'ratio']]
    decile_scores = {tool: [d[tool].mean() for d in deciles] for tool in tool_cols}

    # Prepare decile DataFrame
    decile_df = pd.DataFrame({'Decile': range(1, num_deciles+1)})
    for t, scores in decile_scores.items():
        decile_df[t] = scores
    decile_df = decile_df.round(2)

    # Linear regression per tool
    reg_results = []
    for tool in tool_cols:
        slope, intercept, r_value, p_value, stderr = linregress(decile_df['Decile'], decile_df[tool])
        reg_results.append({
            'Tool': tool,
            'Slope': round(slope, 4),
            'Intercept': round(intercept, 4),
            'R_squared': round(r_value**2, 4),
            'P_value': round(p_value, 4)
        })
    reg_df = pd.DataFrame(reg_results)

    # Plot the F1 trends
    plt.figure(figsize=(12, 8))
    for tool in tool_cols:
        plt.plot(decile_df['Decile'], decile_df[tool], marker='o', label=tool)
    plt.xlabel(f'Decile (1=highest correct:overfitting ratio, {num_deciles}=lowest)')
    plt.ylabel('Average F1 Score')
    plt.title('Detector F1 vs Bug Difficulty Decile')
    plt.legend()
    plt.grid(True)

    if output_file:
        # save plot
        plt.savefig(output_file)
        # save decile CSV
        base = os.path.splitext(os.path.basename(output_file))[0]
        out_dir = os.path.dirname(output_file)
        dec_csv = os.path.join(out_dir, f"{base}_deciles.csv")
        reg_csv = os.path.join(out_dir, f"{base}_regression.csv")
        decile_df.to_csv(dec_csv, index=False)
        reg_df.to_csv(reg_csv, index=False)
        print(f"Decile data saved to {dec_csv}")
        print(f"Regression results saved to {reg_csv}")
    else:
        plt.show()
        print("\nDecile averages:")
        print(decile_df.to_string(index=False))
        print("\nRegression results:")
        print(reg_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process bug data, plot metrics by deciles, and run linear regression.')
    parser.add_argument('csv_path', help='Path to the input CSV file.')
    parser.add_argument('--deciles', type=int, default=9, help='Number of deciles (default 9).')
    parser.add_argument('--output', help='Path to save output plot and CSVs.')
    args = parser.parse_args()
    process_and_plot(args.csv_path, args.deciles, args.output)
