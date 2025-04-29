import pandas as pd
import argparse
import os

def filter_patches(reference_csv, process_csv, output_csv=None):
    """
    Filter process_csv to only include rows with patch_name values found in reference_csv.
    
    Args:
        reference_csv (str): Path to the reference CSV file
        process_csv (str): Path to the CSV file to be filtered
        output_csv (str, optional): Path for the output filtered CSV. If None, will use process_csv with "_filtered" suffix
    
    Returns:
        str: Path to the created filtered CSV file
    """
    # Read the CSV files
    reference_df = pd.read_csv(reference_csv)
    process_df = pd.read_csv(process_csv)
    
    # Get the set of patch names from reference CSV
    reference_patch_names = set(reference_df['patch_name'])
    
    # Filter the process DataFrame to include only rows with patch_name in the reference set
    filtered_df = process_df[process_df['patch_name'].isin(reference_patch_names)]
    
    # Create output filename if not provided
    if output_csv is None:
        base_name = os.path.splitext(process_csv)[0]
        output_csv = f"{base_name}_filtered.csv"
    
    # Write the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"Original CSV had {len(process_df)} rows, filtered CSV has {len(filtered_df)} rows")
    return output_csv

def main():
    parser = argparse.ArgumentParser(description='Filter a CSV file based on patch names from a reference CSV.')
    parser.add_argument('reference_csv', help='Path to the reference CSV file')
    parser.add_argument('process_csv', help='Path to the CSV file to be filtered')
    parser.add_argument('--output', '-o', help='Path for the output filtered CSV')
    
    args = parser.parse_args()
    
    output_path = filter_patches(args.reference_csv, args.process_csv, args.output)
    print(f"Filtered CSV saved to: {output_path}")

if __name__ == "__main__":
    main()
