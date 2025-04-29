# Directory Structure

- `all_patches/`
  - Contains the initial patch dataset before deduplication. All patches have the naming convention: \<project\>\_\<version\>\_\<apr tool\>\_Patch\_\<timestamp\>\_\<...\>.diff
  - Subfolders:
    - `correct/`: Correct patches.
    - `overfitting/`: Overfitting patches.
- `buggy/`
  - Contains buggy versions of projects.
- `fixed/`
  - Contains developer fixed project versions.
- `figure_replication/`
  - Materials needed to replicate all figures in the report.
  - Subfolders:
    - `figures/`: Raw data, tables, and visualisations.
    - `scripts/`: Scripts to generate figures.
    - `README.md`: Instructions to replicate figures.
- `patches_by_time/`
  - Patches grouped by discovery time.
- `results/`
  - Outputs from running evaluation tools.
- `tools/`
  - Tools with modified source code used for benchmarking.
  - `README.md`: Detailed instructions for each tool on how to replicate experimental results.

# Study Replication

1. **Checkout Defects4J Projects**  
   Use the provided script [`checkout_d4j.py`](checkout_d4j.py) to automatically download all necessary buggy and fixed project versions.

   > **Note:** This requires Defects4J to be installed and configured. Defects4J depends on Java 11. You must update the `JAVA_11_PATH` constant in `checkout_d4j.py` to point to a valid Java 11 installation.  
   > For setup instructions, refer to the [Defects4J GitHub repository](https://github.com/rjust/defects4j).

2. **Run Evaluation Tools**  
   Follow the setup and usage instructions provided in [`tools/README.md`](tools/README.md) to run the evaluation tools and move outputs to the `results/` directory as described.

3. **Replicate Figures**  
   After obtaining all experimental outputs, replicate the figures and tables by following the instructions in [`figure_replication/README.md`](figure_replication/README.md).