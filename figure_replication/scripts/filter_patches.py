import pandas as pd
import argparse
import os

SUBFOLDERS = [
    "DL4PatchCorrectness",
    "entropy_delta",
    "FIXCHECK",
    "Invalidator",
    "LLM4PatchCorrectness"
]

def filter_common_patches(results_dir, file_name):
    """
    For each subfolder, filter the file to only include rows with patch_name values found in all files.

    Args:
        results_dir (str): Path to the results directory containing subfolders
        file_name (str): Name of the file to look for in each subfolder

    Returns:
        dict: Mapping from subfolder to output filtered CSV path (for those that exist)
    """
    # Collect patch_name sets from each subfolder's file (if exists)
    patch_sets = []
    file_paths = {}
    for sub in SUBFOLDERS:
        sub_path = os.path.join(results_dir, sub, file_name)
        if os.path.isfile(sub_path):
            df = pd.read_csv(sub_path)
            patch_names = set(df['patch_name'])
            patch_sets.append(patch_names)
            file_paths[sub] = sub_path

    if not patch_sets:
        print("No files found in subfolders.")
        return {}

    # Find intersection of patch_names across all files
    common_patches = set.intersection(*patch_sets)

    # Filter and save each file in its respective subfolder
    output_files = {}
    for sub, path in file_paths.items():
        df = pd.read_csv(path)
        filtered_df = df[df['patch_name'].isin(common_patches)]
        base, ext = os.path.splitext(file_name)
        output_path = os.path.join(results_dir, sub, f"{base}_filtered{ext}")
        filtered_df.to_csv(output_path, index=False)
        output_files[sub] = output_path
        print(f"Processed {sub}: {len(df)} rows -> {len(filtered_df)} rows")

    return output_files

def main():
    parser = argparse.ArgumentParser(description='Filter CSV files in subfolders based on common patch names.')
    parser.add_argument('results_dir', help='Path to the results directory containing subfolders')
    parser.add_argument('file_name', help='Name of the file to look for in each subfolder')
    
    args = parser.parse_args()
    
    output_paths = filter_common_patches(args.results_dir, args.file_name)
    for sub, path in output_paths.items():
        print(f"Filtered CSV for {sub} saved to: {path}")

if __name__ == "__main__":
    main()
