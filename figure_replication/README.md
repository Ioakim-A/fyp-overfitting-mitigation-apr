# Reproducing Figures from the Experiment

This document outlines the steps required to re-create all the figures used in the report. The process involves two main phases:

1. **Data Generation** – Running Python scripts to produce CSV files.
2. **Visualisation** – Using the generated CSVs as input to visualisation scripts.

---

## 🧰 Prerequisites

Before running any scripts, make sure the following are installed:

- Python >= 3.8
- Required packages (install with `pip install -r figure-replication-requirements.txt`)

## 🔄 Step-by-Step Instructions

**Before completing any step, ensure your working directory is `figure_replication/scripts`**

### 0. Filter Patch Predictions (Already Done)

Tools FIXCHECK and Invalidator produced predictions for 798 patches whilst entropy_delta, DL4PatchCorrectness, LLM4PatchCorrectness were able to predict the correctness of an additional 21 non-compilable patches. To ensure comparisons are made for the same 798 patches across all tools, we filter using either FIXCHECK or Invalidator's prediction csv as a reference to filter out patches from other csvs:

```bash
python filter_patches.py <path to reference csv e.g. Invalidator> <path to csv to filter e.g. LLM4PatchCorrectness>
```

For example:
```bash
python filter_patches.py ../../results/Invalidator/8h_deduplicated.csv ../../results/LLM4PatchCorrectness/8h_deduplicated.csv
```

Results will be saved in the target directory and the csv will take the naming convention of the target file, appending '_filtered' e.g. 8h_deduplicated_filtered.csv

**Note that this step has already been completed.**

### 1. Generate CSV Files

These scripts will generate intermediate CSV files needed for plotting.

#### RQ3 Data

```bash
# Generate overall performance data across 798 bugs with Random Selection metric and bootstrapping.
# Note that bootstrapping can take several minutes so if you are not interested in error bars you can exclude --bootstrap
python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --overall --bootstrap --include-wbc --wbc-p-overfit 0.50 --format csv --output ../figures/raw_data/overall.csv 
```

The script above uses the weighted probability classifier with a 50% probability of guessing overfitting, effectively making it equivalent to Random Selection. Therefore, in the generated csv, replace its name 'WBC' to 'RS'.

```bash
# Generate MCC scores on bug-level prediction for each tool
python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --metric 'Smooth MCC' --aggregate bug --format csv --output ../figures/raw_data/mcc_by_bug.csv
```

```bash
# Generate MCC scores on APR-tool-level prediction for each tool
python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --metric 'Smooth MCC' --aggregate approach --format csv --output ../figures/raw_data/mcc_by_apr_tool.csv
```

```bash
# Generate F1 scores on bug-level prediction for each tool
python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --metric 'F1 Score' --aggregate bug --format csv --output ../figures/raw_data/f1.csv
```

#### RQ4 Data

```bash
# Generate RS -85 and -95 baseline csvs
python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --aggregate bug --metric RS --confidence 85 --format csv --output ../figures/tables/rq4/rs85.csv

python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --aggregate bug --metric RS --confidence 95 --format csv --output ../figures/tables/rq4/rs95.csv
```

### 2. Generate Figures

These scripts use the CSVs created in the previous step to generate the final figures.

#### RQ3 Figures

```bash
# Generate bar charts for overall performance data
python produce_bar_chart.py --with-ci --input_csv ../figures/raw_data/overall.csv --output_dir ../figures/visualisations/rq3
```

```bash
# Generate upset plots for correct and overfitting detection
python produce_upset_diagrams.py 8h_deduplicated_filtered --input_dir ../../results --output_dir ../figures/visualisations/rq3
```

```bash
# Generate violin plot for MCC bug-prediction comparison and box plot for MCC project comparison, and print stats used to aid figure
python produce_violin_plots.py --input_csv ../figures/raw_data/mcc_by_bug.csv --output_dir ../figures/visualisations/rq3 --colour_by_project
```  

```bash
# Generate heatmap for APR tool - Overfitting detection tool MCC comparison
python produce_tool_apr_heatmap.py ../figures/raw_data/mcc_by_apr_tool.csv --output ../figures/visualisations/rq3/apr_tool_mcc_heatmap.png
```

```bash
# Generate bug hardness plot and regression stats
python produce_hardness_plot.py ../figures/raw_data/f1.csv --deciles 9 --output ../figures/visualisations/rq3/bug_hardness_f1.png
```

#### RQ4 Figures

```bash
# Generate RS -85 and -95 inspection tables (latex)
python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --aggregate bug --metric 'RS' --confidence 85 --format latex --output ../figures/tables/rq4/rs85-latex.txt

python produce_csvs_or_latex_tables.py 8h_deduplicated_filtered --aggregate bug --metric 'RS' --confidence 95 --format latex --output ../figures/tables/rq4/rs95-latex.txt
```

```bash
# Generate RS -85 and -95 Median and Mean comparison stats (printed to terminal)
python produce_rsb_metrics.py ../figures/tables/rq4/rs85.csv

python produce_rsb_metrics.py ../figures/tables/rq4/rs95.csv
```

```bash
# Generate comparisons for tools vs WBC. For this, we reuse /figures/raw_data/overall.csv by making a copy and deleting the RS row as that is not an APR tool, and saving the result as overall-no-rs.csv. The script also takes the number of correct to overfitting patches e.g. in the case of the 8h_deduplicated_filtered dataset, that is 127:671. This also prints information about tools' confidence intervals intersecting the margin.
python produce_wbc_graphs.py ../figures/raw_data/overall-no-rs.csv 127 671 --out ../figures/visualisations/rq4/tools_vs_wbc.png
```

### 3. Filtering Results by Time
The above results produce comparisons for the 8 hour dataset. Since this contains all patches you can filter the results to e.g. patches generated in 1 hour and generate any of the above figures again.

This can be done by running for a given tool:
```bash
# Repeat for all approaches
python filter_results_by_time.py --from-hour 8 --to-hour 1 --approach LLM4PatchCorrectness
```

Then you will need to keep in mind the non-compilable patches that may be present in Invalidator / FIXCHECK, so you may need to run `filter_patches.py` as described in Step 0 to ensure a fair comparison.