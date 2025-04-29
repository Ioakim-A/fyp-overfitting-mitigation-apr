import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

name_conversion = {
    'entropy_delta':        'Yang et al.',
    'Invalidator':          'Invalidator',
    'FIXCHECK':             'FIXCHECK',
    'LLM4PatchCorrectness': 'LLM4Patch\nCorrect',
    'DL4PatchCorrectness':  'Tian et al.',
}

def create_violin_plot(csv_file, output_dir='.', color_by_project=False):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file).iloc[:-1]
    metrics = df.columns[1:]

    plot_data = []
    for _, row in df.iterrows():
        bug     = row['Bug']
        project = bug.split('-')[0]
        for m in metrics:
            v = row[m]
            if v != 'N/A':
                plot_data.append({
                    'Metric':  name_conversion[m],
                    'Value':   float(v),
                    'Project': project
                })
    plot_df = pd.DataFrame(plot_data)
    
    print("MCC prediction distribution per tool:")
    for tool in plot_df['Metric'].unique():
        sub = plot_df[plot_df['Metric'] == tool]
        neg = (sub['Value'] < 0).sum()
        zero = (sub['Value'] == 0).sum()
        pos = (sub['Value'] > 0).sum()
        print(f"  {tool}: >0 = {pos}, =0 = {zero}, <0 = {neg}")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13
    })

    plt.figure(figsize=(12, 7))
    if color_by_project:
        sns.violinplot(x='Metric', y='Value', data=plot_df, inner=None,
                       color='lightgray', cut=0)
        sns.boxplot(x='Metric', y='Value', data=plot_df,
                    showcaps=False,
                    boxprops={'facecolor':'none','edgecolor':'black'},
                    whiskerprops={'color':'black'},
                    medianprops={'color':'cyan','linewidth':3},
                    showfliers=False, width=0.25)
        sns.stripplot(x='Metric', y='Value', data=plot_df, size=7,
                      hue='Project', dodge=True, jitter=True, alpha=0.7)
        plt.legend(title='Project', bbox_to_anchor=(1.003, 1), loc='upper left')
    else:
        sns.violinplot(x='Metric', y='Value', data=plot_df, inner='quartile',
                       palette='viridis', cut=0)
        sns.stripplot(x='Metric', y='Value', data=plot_df, size=5,
                      color='black', alpha=0.6)

    plt.title('Distribution of MCC Scores per Tool on Bug-Level Predictions', fontweight='bold')
    plt.xlabel('Tool', fontweight='bold')
    plt.ylabel('MCC', fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    violin_path = os.path.join(output_dir, 'metrics_violin_plot.png')
    plt.tight_layout()
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to {violin_path}")
    plt.close()


def create_project_boxplot(csv_file, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file).iloc[:-1]
    metrics = df.columns[1:]

    # Build a DataFrame of all MCC values per project
    plot_data = []
    for _, row in df.iterrows():
        bug     = row['Bug']
        project = bug.split('-')[0]
        for m in metrics:
            v = row[m]
            if v != 'N/A':
                plot_data.append({
                    'Project': project,
                    'Value':   float(v)
                })
    proj_df = pd.DataFrame(plot_data)
    
    stats = proj_df.groupby('Project')['Value'].agg(
        min='min',
        Q1=lambda x: x.quantile(0.25),
        median='median',
        Q3=lambda x: x.quantile(0.75),
        max='max'
    )
    print("Boxplot stats per project:")
    print(stats, "\n")

    # Count unique bugs per project
    df['Project'] = df['Bug'].str.split('-').str[0]
    bug_counts = df.groupby('Project')['Bug'].nunique().to_dict()
    # Create label with count
    proj_df['ProjLabel'] = proj_df['Project'].map(
        lambda p: f"{p}\n({bug_counts.get(p, 0)} Bugs)"
    )

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13
    })

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='ProjLabel', y='Value', data=proj_df,
                showcaps=True,
                boxprops={'facecolor':'lightblue','edgecolor':'black'},
                whiskerprops={'color':'black'},
                medianprops={'color':'red','linewidth':2},
                showfliers=False)
    sns.stripplot(x='ProjLabel', y='Value', data=proj_df,
                  size=5, color='black', alpha=0.5, jitter=True)

    plt.title('Distribution of MCC Scores per Project on Bug-Level Predictions', fontweight='bold')
    plt.xlabel('Project (number of bugs)', fontweight='bold', labelpad=15)
    plt.ylabel('MCC', fontweight='bold')
    plt.xticks(ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    box_path = os.path.join(output_dir, 'project_boxplot.png')
    plt.tight_layout()
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    print(f"Project box plot saved to {box_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate separate MCC plots')
    parser.add_argument('--input_csv', default='tools_metrics.csv')
    parser.add_argument('--output_dir', default='./plots')
    parser.add_argument('--colour_by_project', action='store_true',
                        help='Color violin dots by project')
    args = parser.parse_args()

    create_violin_plot(args.input_csv, args.output_dir, args.colour_by_project)
    create_project_boxplot(args.input_csv, args.output_dir)