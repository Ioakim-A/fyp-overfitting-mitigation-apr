import pandas as pd
import argparse
import re

# Columns to consider for bolding (excluding 'Bug')
value_columns = [
    'entropy_delta', 'Invalidator', 'FIXCHECK',
    'LLM4PatchCorrectness', 'DL4PatchCorrectness', 'RS-85', 'RS-95'
]

# Mapping for renaming columns in the LaTeX table
latex_column_rename = {
    'entropy_delta': 'Yan.',
    'Invalidator': 'Inv.',
    'FIXCHECK': 'FIX.',
    'LLM4PatchCorrectness': 'LLM.',
    'DL4PatchCorrectness': 'Tia.'
}

def bold_min(row):
    vals = row[value_columns]
    min_val = vals.min()
    is_min = vals == min_val
    new_row = row.copy()
    for col in value_columns:
        val = row[col]
        if is_min[col]:
            new_row[col] = f"\\textbf{{{val}}}"
        else:
            new_row[col] = f"{val}"
    return new_row

def main():
    parser = argparse.ArgumentParser(description="Create LaTeX table from CSV with bolded minimum values per row.")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--output", default="output_table.tex", help="Path to output LaTeX file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df.iloc[:-1]  # Exclude the last "Overall" row
    df_bolded = df.apply(bold_min, axis=1)
    # Rename columns for LaTeX output
    df_bolded = df_bolded.rename(columns=latex_column_rename)
    latex_table = df_bolded.to_latex(index=False, escape=False)

    # Cleanup LaTeX formatting
    latex_table = latex_table.replace("nan", "N/A")
    latex_table = latex_table.replace("RS-85", "\\makecell{RS\\\\-85}")
    latex_table = latex_table.replace("RS-95", "\\makecell{RS\\\\-95}")
    latex_table = latex_table.replace("\\textbf", "\\redbold")
    latex_table = latex_table.replace("llllllll", "|l|c|c|c|c|c|c|c|")
    latex_table = latex_table.replace("\\toprule", "\\hline")
    latex_table = latex_table.replace("\\midrule", "\\hline")
    latex_table = latex_table.replace("\\bottomrule", "\\hline")

    for header in ["Bug", "Yan.", "Inv.", "FIX.", "LLM.", "Tia.", "\\makecell{RS\\\\-85}", "\\makecell{RS\\\\-95}"]:
        latex_table = latex_table.replace(header, f"\\textbf{{{header}}}")

    # Add scriptsize to patch counts in first column
    latex_table = re.sub(
        r'(\w+-\d+)\s*\((\d+:\d+)\)',
        lambda m: f"{m.group(1)} \\scriptsize{{({m.group(2)})}}",
        latex_table
    )

    latex_table = latex_table.replace(".0", "")

    with open(args.output, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {args.output}")

if __name__ == "__main__":
    main()
