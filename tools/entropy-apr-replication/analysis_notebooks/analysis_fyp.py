import sys
import os
import json
import pandas as pd
patch_directory = "patches/patches_entropy_fyp_8h_deduplicated"
patches_list = []
for subdir, _, files in os.walk(patch_directory):
    for file in files:
        bug = subdir.split("/")[-1]
        project = subdir.split("/")[-2]
        file_path = os.path.join(subdir, file)
        correct = True
        if "incorrect" in file_path:
            correct = False
        with open(file_path, "r") as f:
            try:
                patches_dict = json.load(f)
            except:
                print(file_path)
                continue
        patched_entropy = patches_dict["patched_entropy"]
        original_entropy = patches_dict["original_entropy"]
        average_patch_entropy = sum(patched_entropy.values()) / len(patched_entropy)
        average_original_entropy = sum(original_entropy.values()) / len(
            original_entropy
        )
        patches_dict["avg_patch_entropy"] = average_patch_entropy
        patches_dict["avg_original_entropy"] = average_original_entropy
        patches_dict["entropy_delta"] = average_original_entropy - average_patch_entropy
        patches_list.append(patches_dict)
df = pd.DataFrame(patches_list)
df["correct"] = df["correct"].astype(bool)
df_count = df[df['correct'] == True]

# count how many unique bug_id is in each project for correct patches
print("rq3 dataset (table 1)")
print(df_count.groupby(['project']).size().reset_index(name='counts'))

## Top 2 and top 1 values
def top_n(df, n, project):
    if not project == "Total":
        df = df[df["project"] == project]
    top_n = len(df[df["correct"] & (df["patch_rank"] <= n)])
    print(f"{project}: top {n} ranking: {top_n}")


df["patch_rank"] = df.groupby(["project", "bug_id"])["entropy_delta"].rank(
    ascending=False
)


top_n(df, 1, "Chart")
top_n(df, 2, "Chart")

top_n(df, 1, "Closure")
top_n(df, 2, "Closure")

top_n(df, 1, "Lang")
top_n(df, 2, "Lang")

top_n(df, 1, "Math")
top_n(df, 2, "Math")

top_n(df, 1, "Mockito")
top_n(df, 2, "Mockito")

top_n(df, 1, "Time")
top_n(df, 2, "Time")

top_n(df, 1, "Total")
top_n(df, 2, "Total")

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="whitegrid")
df_melt = df[["project", "correct", "entropy_delta"]]
df_melt.columns = ["project", "correct_patch", "entropy_delta"]


# sns.catplot(
#     data=df_melt,
#     kind="box",
#     col="correct_patch",
#     x="project",
#     y="entropy_delta",
#     hue="project",
#     sharey=True,
#     height=5,
#     saturation=5,
#     linewidth=1.5,
#     dodge=False,
# )

# change above catplot to side by side boxplot

sns.boxplot(data=df_melt, x="project", y="entropy_delta", hue="correct_patch", palette="colorblind")
plt.xlabel('', fontsize="15")
plt.ylabel('Entropy delta', fontsize="15")
# change font size
plt.xticks(fontsize="15")
plt.yticks(fontsize="15")
plt.savefig("analysis_notebooks/plots/fyp_tested_ranking_edelta.pdf")

def classification(df, prior_tool, cutoff):
    
    df["patch_rank"] = df.groupby(["project", "bug_id"])["entropy_delta"].rank(
        ascending=False
    )
    df_mean_delta = (
        df.groupby(["project", "bug_id", "correct"])["entropy_delta"]
        .mean()
        .reset_index(name="avg_entropy_delta")
    )
    df_binary_delta = (
        df_mean_delta.groupby(["project", "bug_id", "correct"])["avg_entropy_delta"]
        .apply(lambda x: (x > cutoff).sum())
        .reset_index(name="positive_delta")
    )

    true_positive_df = df_binary_delta[
        (df_binary_delta["correct"]) & df_binary_delta["positive_delta"] == 1
    ]
    false_positive_df = df_binary_delta[
        ~(df_binary_delta["correct"]) & (df_binary_delta["positive_delta"] == 1)
    ]
    true_negative_df = df_binary_delta[
        ~(df_binary_delta["correct"]) & (df_binary_delta["positive_delta"] == 0)
    ]
    false_negative_df = df_binary_delta[
        (df_binary_delta["correct"]) & (df_binary_delta["positive_delta"] == 0)
    ]

    # Create CSV output with patch analysis
    df_patches = df.copy()
    df_patches['prediction'] = (df_patches['entropy_delta'] > cutoff).astype(int)
    df_patches['patch_name'] = df_patches['file_path'].apply(lambda x: x.split('/')[-1])
    df_patches['correctness'] = df_patches['correct'].apply(lambda x: 'correct' if x else 'overfitting')
    df_patches['prediction'] = df_patches['prediction'].apply(lambda x: 'correct' if x == 1 else 'overfitting')
    
    # Select and write to CSV
    result_df = df_patches[['patch_name', 'correctness', 'prediction']]
    csv_path = f"entropy_analysis_fyp_cutoff_{cutoff}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Patch analysis written to {csv_path}")
    
    # Continue with existing metrics calculation
    true_positive = len(true_positive_df)
    false_positive = len(false_positive_df)
    true_negative = len(true_negative_df)
    false_negative = len(false_negative_df)

    print(f"True positive count: {true_positive}")
    print(f"True negative count: {true_negative}")

    print(f"False positive count: {false_positive}")
    print(f"False negative count: {false_negative}")

    accuracy = round(
        (true_positive + true_negative)
        / (true_positive + true_negative + false_positive + false_negative),
        3,
    )
    precision = round(true_positive / (true_positive + false_positive), 3)
    pos_recall = round(true_positive / (true_positive + false_negative), 3)
    neg_recall = round(true_negative / (true_negative + false_positive), 3)
    f1 = round(
        true_positive / (true_positive + 0.5 * (false_positive + false_negative)), 3
    )


    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"+recall: {pos_recall}")
    print(f"-recall: {neg_recall}")
    print(f"f1: {f1}")

classification(df, "i", -0.55)