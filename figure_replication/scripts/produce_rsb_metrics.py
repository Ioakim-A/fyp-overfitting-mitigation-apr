import pandas as pd
import re
import argparse

def extract_patch_count(bug_str):
    """Extracts total patches from format like 'Chart-1 (40:105)'."""
    match = re.search(r'\((\d+):(\d+)\)', bug_str)
    if match:
        return int(match.group(1)) + int(match.group(2))
    return None

def compare_rsb_to_tools(csv_path):
    df = pd.read_csv(csv_path)

    # Drop 'Overall' or non-bug rows
    df = df[~df['Bug'].str.lower().str.contains("overall", na=False)]

    # Extract total patch count from Bug column
    df['TotalPatches'] = df['Bug'].apply(extract_patch_count)

    # Tools to compare
    tool_columns = [col for col in df.columns if col not in ['Bug', 'RSB', 'TotalPatches']]

    # Convert tool + RSB columns to numeric, treating 'N/A' as NaN
    for col in tool_columns + ['RSB']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Pairwise RSB <= tool (with NaNs treated as tool failed, RSB wins)
    pairwise_results = {}
    for tool in tool_columns:
        rsb_wins = df[tool].isna() | (df['RSB'] <= df[tool])
        pairwise_results[tool] = rsb_wins.sum()

    # RSB wins across all tools
    rsb_beats_all = df[tool_columns].apply(
        lambda row: all(pd.isna(val) or df.loc[row.name, 'RSB'] <= val for val in row),
        axis=1
    )
    total_all_wins = rsb_beats_all.sum()

    # Fill NaNs with TotalPatches and compute median & mean
    summary_stats = {}
    for col in tool_columns + ['RSB']:
        filled_values = df[col].combine_first(df['TotalPatches'])  # Fill NaNs with per-bug total
        summary_stats[col] = {
            'median': filled_values.median(),
            'mean': filled_values.mean()
        }

    return pairwise_results, total_all_wins, summary_stats

def main():
    parser = argparse.ArgumentParser(description="Compare RSB results to other tools.")
    parser.add_argument('csv_path', type=str, help="Path to the CSV file")
    args = parser.parse_args()

    pairwise_result, all_tools_win_count, stats = compare_rsb_to_tools(args.csv_path)

    confidence_level = args.csv_path.split('/')[-1].replace(".csv", "").replace("rs", "")

    print(f"\nPairwise RSB-{confidence_level} <= tool count:")
    for tool, count in pairwise_result.items():
        print(f"{tool}: {count}")

    print(f"\nRSB-{confidence_level} wins against all tools simultaneously in {all_tools_win_count} cases.")

    print("\nMedian and Mean values (filling NAs with total patches per bug):")
    for tool, s in stats.items():
        print(f"{tool}: median = {s['median']}, mean = {s['mean']:.2f}")

if __name__ == "__main__":
    main()
