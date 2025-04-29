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

To replicate our study, first you need to checkout all the relevant Defects4J projects. This is done automatically by a python script we provide: `checkout_d4j.py`. Note that this requires Defects4J to be set up. Defects4J requires Java 11, and the script provided has a constant **JAVA_11_PATH** that needs to be replaced with the path to a valid Java 11 installation. Detailed instructions to set up Defects4J can be found: https://github.com/rjust/defects4j

The next step is to obtain experimental results by running each tool. This can be done by following the detailed setup and usage instructions for each tool in `tools/README.md`

Once you have all experimental results in the `results` directory, you can analyse the data, replicating the figures in our report by following instructions in `figure_replication/README.md`