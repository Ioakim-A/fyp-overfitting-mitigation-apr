import argparse
import subprocess
import sys
import shutil


def main():
    parser = argparse.ArgumentParser(description="Replication script runner")
    parser.add_argument(
        "dataset_name",
        choices=["petke-8h", "petke-1h", "repairllama", "combined"],
        help="Dataset to use: 'petke-8h', 'petke-1h' or 'repairllama'"
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="If set, skips the RQ3 and RQ4 data processing steps and only generates figures."
    )
    args = parser.parse_args()

    print(f"Selected dataset: {args.dataset_name}")

    if args.dataset_name == "petke-8h":
        output_dir = "../petke-8h_figures/"
        results_csv_name = "8h_deduplicated_filtered"
        num_correct_patches = "127"
        num_overfitting_patches = "671"
    elif args.dataset_name == "petke-1h":
        output_dir = "../petke-1h_figures/"
        results_csv_name = "1h_deduplicated_filtered"
        num_correct_patches = "127"
        num_overfitting_patches = "671"
    elif args.dataset_name == "repairllama":
        output_dir = "../repairllama_figures/"
        results_csv_name = "repairllama_filtered"
        num_correct_patches = "63"
        num_overfitting_patches = "106"
    elif args.dataset_name == "combined":
        output_dir = "../combined_figures/"
        results_csv_name = "combined_filtered"
        num_correct_patches = "190"
        num_overfitting_patches = "777"
    else:
        raise ValueError("Invalid dataset name provided.")

    if not args.figures:
        print("### RQ3 Data Processing ###")
        # Generate overall performance data with Random Selection metric and bootstrapping.
        # Note that bootstrapping can take several minutes so if you are not interested in error bars you can exclude --bootstrap
        print("Generating overall performance data with Random Selection metric and bootstrapping...")
        result = subprocess.run(
            [
                sys.executable,
                "produce_csvs_or_latex_tables.py",
                results_csv_name,
                "--overall",
                "--bootstrap",
                "--include-wbc",
                "--wbc-p-overfit", "0.50",
                "--format", "csv",
                "--output", f"{output_dir}/raw_data/overall.csv"
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Replace WBC with RS in resulting CSV file.
        overall_csv_path = f"{output_dir}/raw_data/overall.csv"
        with open(overall_csv_path, "r") as file:
            content = file.read()
        content = content.replace("WBC", "RS")
        with open(overall_csv_path, "w") as file:
            file.write(content)

        # Generate MCC scores on bug-level prediction for each tool
        print("\n\n\nGenerate MCC scores on bug-level prediction for each tool...")
        result = subprocess.run(
            [
                sys.executable,
                "produce_csvs_or_latex_tables.py",
                results_csv_name,
                "--metric", "Smooth MCC",
                "--aggregate", "bug",
                "--format", "csv",
                "--output", f"{output_dir}/raw_data/mcc_by_bug.csv"
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Generate MCC scores on APR-tool-level prediction for each tool
        print("\n\n\nGenerate MCC scores aggregated by APR tool...")
        result = subprocess.run(
            [
                sys.executable,
                "produce_csvs_or_latex_tables.py",
                results_csv_name,
                "--metric", "Smooth MCC",
                "--aggregate", "approach",
                "--format", "csv",
                "--output", f"{output_dir}/raw_data/mcc_by_apr_tool.csv"
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Generate F1 scores on bug-level prediction for each tool
        print("\n\n\nGenerate F1 scores on bug-level prediction for each tool...")
        result = subprocess.run(
            [
                sys.executable,
                "produce_csvs_or_latex_tables.py",
                results_csv_name,
                "--metric", "F1 Score",
                "--aggregate", "bug",
                "--format", "csv",
                "--output", f"{output_dir}/raw_data/f1.csv"
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print("### RQ3 Data Processing Completed ###")
        print("\n\n### RQ4 Data Processing ###")

        # Generate RS -85 and -95 baseline csvs
        print("Generating RS -85 and -95 baseline csvs...")
        result = subprocess.run(
            [
                sys.executable,
                "produce_csvs_or_latex_tables.py",
                results_csv_name,
                "--aggregate", "bug",
                "--metric", "RS",
                "--confidence", "85",
                "--format", "csv",
                "--output", f"{output_dir}/tables/rq4/rs85.csv"
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        result = subprocess.run(
            [
                sys.executable,
                "produce_csvs_or_latex_tables.py",
                results_csv_name,
                "--aggregate", "bug",
                "--metric", "RS",
                "--confidence", "95",
                "--format", "csv",
                "--output", f"{output_dir}/tables/rq4/rs95.csv"
            ],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print("### RQ4 Data Processing Completed ###")

    print("\n\n### RQ3 Figure Generation ###")

    # Generate bar charts for overall performance data
    print("Generating bar charts for overall performance data...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_bar_chart.py",
            "--with-ci",
            "--input_csv", f"{output_dir}/raw_data/overall.csv",
            "--output_dir", f"{output_dir}/visualisations/rq3"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Generate upset plots for correct and overfitting detection
    print("Generating upset plots for correct and overfitting detection...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_upset_diagrams.py",
            results_csv_name,
            "--input_dir", "../../results",
            "--output_dir", f"{output_dir}/visualisations/rq3"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Generate violin plot for MCC bug-prediction comparison and box plot for MCC project comparison, and print stats used to aid figure
    print("Generating violin plot for MCC bug-prediction comparison and box plot for MCC project comparison...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_violin_plots.py",
            "--input_csv", f"{output_dir}/raw_data/mcc_by_bug.csv",
            "--output_dir", f"{output_dir}/visualisations/rq3",
            "--colour_by_project"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Generate heatmap for APR tool - Overfitting detection tool MCC comparison
    print("Generating heatmap for APR tool - Overfitting detection tool MCC comparison...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_tool_apr_heatmap.py",
            f"{output_dir}/raw_data/mcc_by_apr_tool.csv",
            "--output",
            f"{output_dir}/visualisations/rq3/apr_tool_mcc_heatmap.png"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Generate bug hardness plot and regression stats
    print("Generating bug hardness plot and regression stats...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_hardness_plot.py",
            f"{output_dir}/raw_data/f1.csv",
            "--deciles", "9",
            "--output", f"{output_dir}/visualisations/rq3/bug_hardness_f1.png"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    print("### RQ3 Figure Generation Completed ###")
    print("\n\n### RQ4 Figure Generation ###")

    # Generate RS -85 and -95 inspection tables (latex)
    print("Generating RS -85 and -95 inspection tables (latex)...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_csvs_or_latex_tables.py",
            results_csv_name,
            "--aggregate", "bug",
            "--metric", "RS",
            "--confidence", "85",
            "--format", "latex",
            "--output", f"{output_dir}/tables/rq4/rs85-latex.txt"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    result = subprocess.run(
        [
            sys.executable,
            "produce_csvs_or_latex_tables.py",
            results_csv_name,
            "--aggregate", "bug",
            "--metric", "RS",
            "--confidence", "95",
            "--format", "latex",
            "--output", f"{output_dir}/tables/rq4/rs95-latex.txt"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Generate RS -85 and -95 Median and Mean comparison stats (printed to terminal)
    print("Generating RS -85 and -95 Median and Mean comparison stats...")
    result = subprocess.run(
        [
            sys.executable,
            "produce_rsb_metrics.py",
            f"{output_dir}/tables/rq4/rs85.csv"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    result = subprocess.run(
        [
            sys.executable,
            "produce_rsb_metrics.py",
            f"{output_dir}/tables/rq4/rs95.csv"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Generate comparisons for tools vs WBC. 
    print("Generating comparisons for tools vs WBC...")
    # Copy overall.csv to overall-no-rs.csv
    overall_csv_path = f"{output_dir}/raw_data/overall.csv"
    overall_no_rs_csv_path = f"{output_dir}/raw_data/overall-no-rs.csv"
    shutil.copyfile(overall_csv_path, overall_no_rs_csv_path)

    # Remove the row that starts with RS
    with open(overall_no_rs_csv_path, "r") as infile:
        lines = infile.readlines()
    with open(overall_no_rs_csv_path, "w") as outfile:
        for line in lines:
            if not line.startswith("RS,"):
                outfile.write(line)

    result = subprocess.run(
        [
            sys.executable,
            "produce_wbc_graphs.py",
            f"{output_dir}/raw_data/overall-no-rs.csv",
            num_correct_patches,
            num_overfitting_patches,
            "--out",
            f"{output_dir}/visualisations/rq4/tools_vs_wbc.png"
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


if __name__ == "__main__":
    main()
