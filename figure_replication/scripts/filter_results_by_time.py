import os
import sys
import argparse
import csv

def read_patch_contents(base_dir):
    """
    Recursively read all .diff files under base_dir and return a mapping filename -> content.
    """
    mapping = {}
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith('.diff'):
                path = os.path.join(root, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        mapping[fname] = f.read()
                except Exception as e:
                    print(f"Warning: could not read {path}: {e}")
    return mapping


def extract_tool(filename):
    """
    Extract the third underscore-separated segment as the tool name.
    """
    parts = filename.split('_')
    if len(parts) >= 4:
        return parts[2]
    return None


def build_mapping(target_dir, source_dir):
    """
    Build a mapping from target filenames to source filenames based on file content,
    allowing duplicate content only if originating tools differ.
    Also prints how many patches were found in the target directory.
    """
    target = read_patch_contents(target_dir)
    print(f"Found {len(target)} patches in '{target_dir}'")
    source = read_patch_contents(source_dir)

    # Map (tool, content) -> source filename
    content_tool_map = {}
    for fname, content in source.items():
        tool = extract_tool(fname)
        key = (tool, content)
        if key in content_tool_map:
            prev = content_tool_map[key]
            print(f"Warning: duplicate content for tool '{tool}' in source for {prev} and {fname}")
        else:
            content_tool_map[key] = fname

    mapping = {}
    for t_fname, t_content in target.items():
        t_tool = extract_tool(t_fname)
        # direct filename match
        if t_fname in source:
            mapping[t_fname] = t_fname
        else:
            key = (t_tool, t_content)
            if key in content_tool_map:
                mapping[t_fname] = content_tool_map[key]
            else:
                print(f"Error: no match found in source for {t_fname} (tool={t_tool})")
    return mapping


def filter_predictions(input_csv, mapping, output_csv):
    """
    Read input_csv with columns: patch_name,correctness,prediction.
    Remap patch_name from source to target based on mapping and write to output_csv.
    Rows with no mapping are silently skipped.
    """
    # Invert mapping: source_name -> target_name
    inv_map = {src: tgt for tgt, src in mapping.items()}

    with open(input_csv, 'r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = []
        for row in reader:
            src_name = row['patch_name']
            if src_name in inv_map:
                tgt_name = inv_map[src_name]
                row['patch_name'] = tgt_name
                rows.append(row)
            # otherwise: skip patch not in target mapping

    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Remap and filter prediction CSVs across deduplicated patch sets by hour.')
    parser.add_argument('--from-hour', type=int, required=True,
                        help='Source deduplication hour (e.g., 8)')
    parser.add_argument('--to-hour', type=int, required=True,
                        help='Target deduplication hour (e.g., 1)')
    parser.add_argument('--approach', required=True,
                        help='Name of the approach folder under results')
    args = parser.parse_args()
    
    patches_base_dir = '../../patches_by_time'
    results_base_dir = '../../results'

    source_dir = os.path.join(patches_base_dir, f"patches_{args.from_hour}h_deduplicated")
    target_dir = os.path.join(patches_base_dir, f"patches_{args.to_hour}h_deduplicated")
    input_csv = os.path.join(results_base_dir, args.approach,
                             f"{args.from_hour}h_deduplicated.csv")
    output_csv = os.path.join(results_base_dir, args.approach,
                              f"{args.to_hour}h_deduplicated.csv")

    print(f"Filtering from {args.from_hour}h to {args.to_hour}h for approach '{args.approach}'")
    mapping = build_mapping(target_dir, source_dir)
    filter_predictions(input_csv, mapping, output_csv)
    print(f"Filtered predictions written to {output_csv}")


if __name__ == '__main__':
    main()
