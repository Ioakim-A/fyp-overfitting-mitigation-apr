import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import math
import csv
import numpy as np

RESULTS_DIR = "../../results"

def cluster_bootstrap_ci(df, metric_name, n_bootstrap=5000, ci=95.0):
    """
    Perform cluster-aware bootstrap over bug-tool clusters to estimate CI for a metric.
    df must contain columns: 'patch_name', 'correctness', 'prediction'.

    Clusters are defined by bug-tool combos extracted from patch_name: 
    bug = parts[0] + '-' + parts[1];
    tool = parts[2];
    cluster_id = f"{bug}_{tool}".

    Returns (lower, upper) bounds of the CI.
    """
    # Derive cluster IDs
    def extract_cluster(name):
        parts = name.split('_')
        if len(parts) >= 3:
            bug = f"{parts[0]}-{parts[1]}"
            tool = parts[2]
            return f"{bug}_{tool}"
        return name  # fallback

    df = df.copy()
    df['cluster'] = df['patch_name'].apply(extract_cluster)
    clusters = df['cluster'].unique()
    n_clusters = len(clusters)
    scores = []

    for _ in range(n_bootstrap):
        # Sample clusters with replacement
        sampled = np.random.choice(clusters, size=n_clusters, replace=True)
        # Aggregate all patches from sampled clusters
        sample_df = pd.concat([df[df['cluster'] == c] for c in sampled], axis=0)
        y_true = sample_df['correctness'].tolist()
        y_pred = sample_df['prediction'].tolist()
        # Compute metric
        score = calculate_metrics(y_true, y_pred, metric_name)
        scores.append(score)

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper

def calculate_metrics(y_true, y_pred, metric_name):
    """Calculate specified metric from true and predicted labels."""
    # Convert labels to binary (1 for 'correct', 0 for 'overfitting')
    y_true_binary = [1 if label == 'correct' else 0 for label in y_true]
    y_pred_binary = [1 if label == 'correct' else 0 for label in y_pred]
    
    if metric_name == 'Accuracy':
        return accuracy_score(y_true_binary, y_pred_binary)
    elif metric_name == 'Precision':
        return precision_score(y_true_binary, y_pred_binary, zero_division=0)
    elif metric_name == 'Positive Recall':
        return recall_score(y_true_binary, y_pred_binary, zero_division=0)
    elif metric_name == 'Negative Recall':
        # Calculating recall for negative class (overfitting)
        return recall_score([1-y for y in y_true_binary], [1-y for y in y_pred_binary], zero_division=0)
    elif metric_name == 'Balanced Accuracy':
        # Calculating balanced accuracy as average of positive and negative recall
        pos_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        neg_recall = recall_score([1-y for y in y_true_binary], [1-y for y in y_pred_binary], zero_division=0)
        return (pos_recall + neg_recall) / 2
    elif metric_name == 'F1 Score':
        return f1_score(y_true_binary, y_pred_binary, zero_division=0)
    elif metric_name == 'Smooth MCC':
        # Calculate confusion matrix values
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        if tp == 0:
            tp = 1e-12
        if tn == 0:
            tn = 1e-12
        if fp == 0:
            fp = 1e-12
        if fn == 0:
            fn = 1e-12
        
        # Calculate MCC with smoothed values
        numerator = tp * tn - fp * fn
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        return numerator / denominator
    elif metric_name == 'MCC':
        # Check if we have all required components for MCC calculation
        # MCC is undefined if any of the sums in the denominator is zero
        # (which happens when predictions or true values are all the same class)
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        # Check if any of the confusion matrix components is missing
        if (tp + fp) == 0 or (tp + fn) == 0 or (tn + fp) == 0 or (tn + fn) == 0:
            return float('nan')  # This will be displayed as N/A in the table
        
        return matthews_corrcoef(y_true_binary, y_pred_binary)
    elif metric_name == 'Epsilon MCC':
        # MCC using the calculus-based epsilon smoothing for corner cases
        epsilon = 1e-12  # Very small number to replace 0s
        
        tp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)

        # Handle 1-row or 1-column matrices where MCC is perfectly defined
        if (tp > 0 and fp == fn == tn == 0):
            return 1.0
        if (tn > 0 and tp == fp == fn == 0):
            return 1.0
        if (fp > 0 and tp == fn == tn == 0):
            return -1.0
        if (fn > 0 and tp == fp == tn == 0):
            return -1.0

        # If denominator would be zero, apply calculus-style epsilon smoothing
        if (tp + fp == 0) or (tp + fn == 0) or (tn + fp == 0) or (tn + fn == 0):
            a = tp + fn  # total actual positives
            b = fp + tn  # total actual negatives

            if a == 0 or b == 0:
                return 0.0  # Degenerate case: all samples are from one class

            return 0.0  # As per the paper, assign 0 when denominator becomes undefined

        # Otherwise, calculate MCC normally
        numerator = tp * tn - fp * fn
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator

    elif metric_name == 'RS':
        # For RS metric: count TP and FP
        tp = 0
        fp = 0
        for true, pred in zip(y_true, y_pred):
            if pred == 'correct':  # If predicted as correct
                if true == 'correct':  # True positive
                    tp += 1
                else:  # False positive
                    fp += 1
        
        if tp > 0:
            return fp + 1  # Return FP + 1 if at least one correct patch is found
        else:
            return float('nan')  # Return NaN to represent N/A
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

def calculate_rsb(correct_count, total_count, confidence=95.0):
    """
    Calculate Random Sampling Baseline (RSB) - the number of patches that need to be
    sampled at random to have the specified confidence of selecting at least 1 correct patch.
    
    Args:
        correct_count (int): Number of correct patches
        total_count (int): Total number of patches
        confidence (float): Confidence percentage (default: 95.0)
    
    Returns:
        int or float: The RSB value, or NaN if there are no correct patches
    """
    if correct_count <= 0 or total_count <= 0:
        return float('nan')  # No correct patches or no patches at all
    
    # Convert confidence percentage to probability
    target_prob = confidence / 100.0
    
    # Calculate the minimum number of samples needed
    n = 1
    while n <= total_count:
        # Calculate probability using the provided formula
        # Pr(X ≥ 1) = 1 - Pr(X = 0) = 1 - (N-K choose n)/(N choose n)
        numerator = math.comb(total_count - correct_count, n)
        denominator = math.comb(total_count, n)
        
        probability = 1 - (numerator / denominator) if denominator > 0 else 0
        
        if probability >= target_prob:
            return n
        
        n += 1
    
    return total_count  # If we can't achieve the target confidence

def expected_metrics(p_true_correct, p_pred_correct=0.50):
    """
    Computes expected classification metrics for a baseline that guesses 'correct' with p_pred_correct
    and 'overfitting' with 1 - p_pred_correct.
    
    Args:
        p_true_correct (float): Proportion of true labels that are 'correct'.
        p_pred_correct (float): Probability baseline predicts 'correct'. Default is 0.50.
    
    Returns:
        dict: Expected values for Accuracy, Precision, Positive Recall, Negative Recall, F1 Score, MCC.
    """
    a = p_true_correct
    p = p_pred_correct
    q = 1 - p
    b = 1 - a

    TP = p * a
    FN = q * a
    FP = p * b
    TN = q * b

    # Avoid division by zero by using small epsilon where needed
    epsilon = 1e-12

    accuracy = TP + TN
    precision = TP / (TP + FP + epsilon)
    pos_recall = TP / (TP + FN + epsilon)
    neg_recall = TN / (TN + FP + epsilon)
    f1_score = (2 * precision * pos_recall) / (precision + pos_recall + epsilon)

    numerator = TP * TN - FP * FN
    denominator = math.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + epsilon
    )
    mcc = numerator / denominator

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Positive Recall": round(pos_recall, 4),
        "Negative Recall": round(neg_recall, 4),
        "Balanced Accuracy": round((pos_recall + neg_recall) / 2, 4),
        "F1 Score": round(f1_score, 4),
        "MCC": round(mcc, 4)
    }

def generate_csv_output(experiment_type, aggregate_by='approach', metric=None, include_wbc=False, wbc_p_overfit=0.65, confidence=95.0):
    """
    Generate a CSV file for the given experiment type with a single metric.
    
    Args:
        experiment_type (str): The type of experiment to analyze.
        aggregate_by (str): Either 'approach', 'bug', or 'project' to determine how to group results.
        metric (str): The single metric to calculate.
        include_wbc (bool): Whether to include the WBC baseline.
        wbc_p_overfit (float): Probability that WBC predicts overfitting (1 - probability of predicting correct).
        confidence (float): Confidence percentage for RS metric (default: 95.0).
    
    Returns:
        list: A list of rows (each row is a list of values) that can be written to a CSV file.
    """
    if metric is None:
        metric = 'Accuracy'
    
    # Check if RS metric is selected
    has_rs_metric = metric == 'RS'
    if has_rs_metric:
        # If RS is selected, WBC cannot be used
        include_wbc = False
    
    tool_dfs = {}
    for tool_dir in ['entropy_delta', 'Invalidator', 'FIXCHECK', 'LLM4PatchCorrectness', 'DL4PatchCorrectness']:
       tool_path = os.path.join(RESULTS_DIR, tool_dir)
       if os.path.isdir(tool_path):
           csv_file = os.path.join(tool_path, f"{experiment_type}.csv")
           if os.path.exists(csv_file):
               df = pd.read_csv(csv_file)
               
               # Extract project, bug, and approach information from patch_name
               df['project'] = df['patch_name'].apply(lambda x: x.split('_')[0])
               df['bug'] = df['patch_name'].apply(lambda x: f"{x.split('_')[0]}-{x.split('_')[1]}")
               df['approach'] = df['patch_name'].apply(lambda x: x.split('_')[2])
               
               tool_dfs[tool_dir] = df
    
    # Calculate metrics for each tool
    results = {}
    for tool_name, df in tool_dfs.items():
        tool_results = {}
        # Use the extracted columns for grouping
        group_key = 'approach' if aggregate_by == 'approach' else ('bug' if aggregate_by == 'bug' else 'project')
        grouped = df.groupby(group_key)
        
        for group_name, group_df in grouped:
            # Get all true labels and predicted labels for this group
            y_true = group_df['correctness'].tolist()
            y_pred = group_df['prediction'].tolist()
            
            tool_results[group_name] = {metric: calculate_metrics(y_true, y_pred, metric)}
        
        results[tool_name] = tool_results

    # First, gather all groups and calculate proportion of correct patches for each
    all_group_data = {}
    tool_name = list(tool_dfs.keys())[0]
    df = tool_dfs[tool_name]
    # Use the same group_key as above
    group_key = 'approach' if aggregate_by == 'approach' else ('bug' if aggregate_by == 'bug' else 'project')
    grouped = df.groupby(group_key)
    
    for group_name, group_df in grouped:
        if group_name not in all_group_data:
            all_group_data[group_name] = {'correct': 0, 'total': 0}
        
        # Count correct and total patches
        correct_count = sum(group_df['correctness'] == 'correct')
        total_count = len(group_df)
        
        all_group_data[group_name]['correct'] += correct_count
        all_group_data[group_name]['total'] += total_count
    
    # If WBC is enabled, add it to results
    if include_wbc:
        # Calculate WBC metrics for each group
        wbc_results = {}
        for group_name, counts in all_group_data.items():
            p_true_correct = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            # Use 1 - wbc_p_overfit as the probability of predicting correct
            expected_metrics_values = expected_metrics(p_true_correct, 1 - wbc_p_overfit)
            wbc_results[group_name] = {metric: expected_metrics_values.get(metric, float('nan'))}
        
        # Add WBC to results
        results['WBC'] = wbc_results
    
    # If RS metric is selected, calculate RSB for each group
    rsb_results = {}
    if has_rs_metric:
        for group_name, counts in all_group_data.items():
            correct_count = counts['correct']
            total_count = counts['total']
            rsb_value = calculate_rsb(correct_count, total_count, confidence)
            rsb_results[group_name] = {metric: rsb_value}
        
        # Add RSB as a separate column in results
        results['RSB'] = rsb_results
    # Get all unique group names across all tools
    if aggregate_by == 'bug':
        # Custom sorting for bugs: sort by project name alphabetically first, then by bug number numerically
        def bug_sort_key(bug_id):
            parts = bug_id.split('-')
            if len(parts) == 2:
                try:
                    return (parts[0], int(parts[1]))  # Sort by project name, then bug number as integer
                except ValueError:
                    pass
            return bug_id  # Fallback to original string if format doesn't match
        
        all_groups = sorted(all_group_data.keys(), key=bug_sort_key)
    else:
        # For other aggregation types, use standard alphabetical sorting
        all_groups = sorted(all_group_data.keys())
    
    # If RS metric is selected, only include groups with at least one correct patch
    if has_rs_metric:
        all_groups = [group for group in all_groups if all_group_data[group]['correct'] > 0]
    
    # Create CSV content
    csv_rows = []
    
    # Header row with tool names
    header_row = [aggregate_by.capitalize()]
    header_row.extend(results.keys())
    csv_rows.append(header_row)
    
    # Add rows for each group
    for group in all_groups:
        # Get the ratio for this group
        correct_count = all_group_data[group]['correct']
        overfitting_count = all_group_data[group]['total'] - correct_count
        group_label = f"{group} ({correct_count}:{overfitting_count})"
        
        row = [group_label]
        
        # Add values for each tool
        for tool in results.keys():
            if group in results[tool] and metric in results[tool][group]:
                value = results[tool][group][metric]
                if math.isnan(value):
                    row.append("N/A")
                else:
                    row.append(f"{value:.2f}")
            else:
                row.append("N/A")
        
        csv_rows.append(row)
    
    # Add Overall row with averages
    overall_row = ["Overall"]
    
    # Calculate averages for each tool
    for tool in results.keys():
        values = []
        for group in all_groups:
            if group in results[tool] and metric in results[tool][group]:
                value = results[tool][group][metric]
                if not math.isnan(value):
                    values.append(value)
        
        if values:
            overall_row.append(f"{sum(values) / len(values):.2f}")
        else:
            overall_row.append("N/A")
    
    csv_rows.append(overall_row)
    
    return csv_rows

def generate_overall_output(experiment_type, include_wbc=False, wbc_p_overfit=0.65, confidence=95.0, bootstrap=False):
    """
    Generate a comprehensive comparison of all tools across all metrics.
    
    Args:
        experiment_type (str): The type of experiment to analyze.
        include_wbc (bool): Whether to include the WBC baseline.
        wbc_p_overfit (float): Probability that WBC predicts overfitting (default: 0.65).
        confidence (float): Confidence percentage for RS metric.
    
    Returns:
        list: A list of rows where each tool is compared across all metrics.
    """
    metrics = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Positive Recall', 'Negative Recall', 'F1 Score', 'MCC']
    
    # Dictionary to store overall values for each tool and metric
    tool_metrics = {}
    
    # Load data for each tool without grouping
    tool_dfs = {}
    for tool_dir in ['entropy_delta', 'Invalidator', 'FIXCHECK', 'LLM4PatchCorrectness', 'DL4PatchCorrectness']:
        tool_path = os.path.join(RESULTS_DIR, tool_dir)
        if os.path.isdir(tool_path):
            csv_file = os.path.join(tool_path, f"{experiment_type}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                tool_dfs[tool_dir] = df

    # Calculate metrics for each tool
    for tool_name, df in tool_dfs.items():
        y_true = df['correctness'].tolist()
        y_pred = df['prediction'].tolist()
        
        tool_metrics[tool_name] = {}
        for metric in metrics:
            tool_metrics[tool_name][metric] = calculate_metrics(y_true, y_pred, metric)
    
    # Add WBC baseline if requested
    if include_wbc:
        # Calculate proportion of correct patches overall
        all_true = []
        for df in tool_dfs.values():
            all_true.extend(df['correctness'].tolist())
        
        if all_true:
            p_true_correct = all_true.count('correct') / len(all_true)
            expected_metrics_values = expected_metrics(p_true_correct, 1 - wbc_p_overfit)
            
            tool_metrics['WBC'] = {}
            for metric in metrics:
                tool_metrics['WBC'][metric] = expected_metrics_values.get(metric, float('nan'))
                
    if bootstrap:
        for tool, df in tool_dfs.items():
            for m in metrics:
                pt = tool_metrics[tool][m]
                # Skip WBC and NaNs
                if tool != 'WBC' and not math.isnan(pt):
                    low, high = cluster_bootstrap_ci(df, m)
                    tool_metrics[tool][m] = f"{pt:.2f} ({low:.2f}-{high:.2f})"
                else:
                    # Format baseline or missing
                    tool_metrics[tool][m] = f"{pt:.2f}" if not math.isnan(pt) else "N/A"
    
    # Create CSV content
    csv_rows = []
    
    # Header row with metric names
    header_row = ['Tool'] + metrics
    csv_rows.append(header_row)
    
    # Add rows for each tool
    for tool in sorted(tool_metrics.keys()):
        row = [tool]
        
        # Add values for each metric
        for metric in metrics:
            value = tool_metrics[tool].get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.2f}")
            else:
                row.append(value)
        
        csv_rows.append(row)
    
    return csv_rows

def generate_latex_output(experiment_type, aggregate_by, metric, data_rows, use_longtable=False):
    """
    Generate LaTeX table code from the experiment results.
    
    Args:
        experiment_type (str): The type of experiment being analyzed.
        aggregate_by (str): How data is aggregated ('approach', 'bug', or 'project').
        metric (str): The metric being displayed.
        data_rows (list): List of rows, where each row is a list of values.
        use_longtable (bool): Whether to use a longtable environment.
    
    Returns:
        str: LaTeX table code.
    """
    # Extract column headers (tool names)
    headers = data_rows[0][1:]
    
    # Get number of columns for multicolumn command
    num_cols = len(headers) + 1
    
    # Start building LaTeX code
    latex_code = []
    
    # Check if using RS metric (for bolding logic)
    is_rs_metric = metric == 'RS'
    
    if use_longtable:
        latex_code.extend([
            "\\clearpage",
            "\\begin{center}",
            "\\setlength\\LTleft{0pt}",
            "\\setlength\\LTright{0pt}",
            "",
            f"\\begin{{longtable}}{{|l|{'c|' * len(headers)}}}",
            f"\\caption{{{metric} for {experiment_type} experiment (aggregated by {aggregate_by})}}",
            "\\captionsetup{width=\\textwidth}",
            f"\\label{{tab:{experiment_type}_{aggregate_by}}} \\\\",
            "\\hline"
        ])
        
        # Header row
        header_row = f"\\textbf{{{aggregate_by.capitalize()}}}"
        for tool in headers:
            if tool == "entropy_delta":
                header_row += " & \\textbf{Yang et al.}"
            elif tool == "LLM4PatchCorrectness":
                header_row += " & \\textbf{\\makecell{LLM4Patch\\\\Correctness}}"
            elif tool == "DL4PatchCorrectness":
                header_row += " & \\textbf{Tian et al.}"
            else:
                header_row += f" & \\textbf{{{tool}}}"
        header_row += " \\\\"
        
        latex_code.extend([
            header_row,
            "\\hline",
            "\\endfirsthead",
            "",
            "\\hline",
            header_row,  # Same header for continuation pages
            "\\hline",
            "\\endhead",
            "",
            "\\hline",
            f"\\multicolumn{{{num_cols}}}{{r}}{{\\textit{{*Continued on next page}}}} \\\\",
            "\\endfoot",
            "",
            "\\hline",
            "\\endlastfoot",
            ""
        ])
    else:
        latex_code.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{metric} for {experiment_type} experiment (aggregated by {aggregate_by})}}",
            f"\\label{{tab:{experiment_type}_{aggregate_by}}}",
            f"\\begin{{tabular}}{{|l|{'c|' * len(headers)}}}",
            "\\hline",
        ])
        
        # Header row
        header_row = f"\\textbf{{{aggregate_by.capitalize()}}}"
        for tool in headers:
            if tool == "entropy_delta":
                header_row += " & \\textbf{Yang et al.}"
            elif tool == "LLM4PatchCorrectness":
                header_row += " & \\textbf{\\makecell{LLM4Patch\\\\Correctness}}"
            elif tool == "DL4PatchCorrectness":
                header_row += " & \\textbf{Tian et al.}"
            else:
                header_row += f" & \\textbf{{{tool}}}"
        
        latex_code.extend([
            header_row + " \\\\",
            "\\hline"
        ])
    
    # Process data rows (skip the header row)
    for row in data_rows[1:]:
        if row[0] == "Overall":
            # Add a horizontal line before the Overall row
            latex_code.append("\\hline")
            latex_row = f"\\textbf{{{row[0]}}}"
        else:
            # Extract the group name and the ratio from each row
            parts = row[0].split(" (")
            if len(parts) == 2:
                group_name = parts[0]
                ratio = parts[1].rstrip(')')
                latex_row = f"{group_name} \\scriptsize{{({ratio})}}"
            else:
                latex_row = row[0]
        
        # Find the best value in this row to bold it
        # For RS metric, the LOWEST value is best; for others, the HIGHEST value is best
        if is_rs_metric:
            min_val = float('inf')
            min_indices = []
        else:
            max_val = -float('inf')
            max_indices = []
        
        # Skip the first column (group name) when finding the best value
        for i in range(1, len(row)):
            val_str = row[i]
            if val_str != "N/A":
                try:
                    val = float(val_str)
                    if is_rs_metric:
                        # For RS metric, lower is better
                        if val < min_val:
                            min_val = val
                            min_indices = [i]
                        elif val == min_val:
                            min_indices.append(i)
                    else:
                        # For all other metrics, higher is better
                        if val > max_val:
                            max_val = val
                            max_indices = [i]
                        elif val == max_val:
                            max_indices.append(i)
                except ValueError:
                    pass
        
        # Add values to the row, bolding the appropriate values based on the metric
        for i in range(1, len(row)):
            val = row[i]
            best_indices = min_indices if is_rs_metric else max_indices
            
            if i in best_indices and val != "N/A" and row[0] != "Overall":
                latex_row += f" & \\textbf{{{val}}}"
            else:
                latex_row += f" & {val}"
        
        latex_code.append(latex_row + " \\\\")
    
    # End the table
    if use_longtable:
        latex_code.extend([
            "\\end{longtable}",
            "\\end{center}"
        ])
    else:
        latex_code.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])
    
    return "\n".join(latex_code)

def generate_overall_latex(experiment_type, aggregate_by, data_rows, use_longtable=False):
    """
    Generate LaTeX table code for the overall metrics comparison.
    
    Args:
        experiment_type (str): The type of experiment being analyzed.
        aggregate_by (str): How data is aggregated.
        data_rows (list): List of rows, where each row is a list of values.
        use_longtable (bool): Whether to use a longtable environment.
    
    Returns:
        str: LaTeX table code.
    """
    # Extract metrics headers
    metrics = data_rows[0][1:]
    
    # Get number of columns for multicolumn command
    num_cols = len(metrics) + 1
    
    # Start building LaTeX code
    latex_code = []
    
    if use_longtable:
        latex_code.extend([
            "\\clearpage",
            "\\begin{center}",
            "\\setlength\\LTleft{0pt}",
            "\\setlength\\LTright{0pt}",
            "",
            f"\\begin{{longtable}}{{|l|{'c|' * len(metrics)}}}",
            f"\\caption{{Overall metrics comparison for {experiment_type} experiment (aggregated by {aggregate_by})}}",
            "\\captionsetup{width=\\textwidth}",
            f"\\label{{tab:{experiment_type}_{aggregate_by}_overall}} \\\\",
            "\\hline"
        ])
        
        # Header row
        header_row = "\\textbf{Tool}"
        for metric in metrics:
            header_row += f" & \\textbf{{{metric}}}"
        header_row += " \\\\"
        
        latex_code.extend([
            header_row,
            "\\hline",
            "\\endfirsthead",
            "",
            "\\hline",
            header_row,  # Same header for continuation pages
            "\\hline",
            "\\endhead",
            "",
            "\\hline",
            f"\\multicolumn{{{num_cols}}}{{r}}{{\\textit{{*Continued on next page}}}} \\\\",
            "\\endfoot",
            "",
            "\\hline",
            "\\endlastfoot",
            ""
        ])
    else:
        latex_code.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Overall metrics comparison for {experiment_type} experiment (aggregated by {aggregate_by})}}",
            f"\\label{{tab:{experiment_type}_{aggregate_by}_overall}}",
            f"\\begin{{tabular}}{{|l|{'c|' * len(metrics)}}}",
            "\\hline",
        ])
        
        # Header row
        header_row = "\\textbf{Tool}"
        for metric in metrics:
            header_row += f" & \\textbf{{{metric}}}"
        
        latex_code.extend([
            header_row + " \\\\",
            "\\hline"
        ])
    
    # Process data rows (skip the header row)
    for row in data_rows[1:]:
        tool_name = row[0]
        # Format tool names nicely
        if tool_name == "entropy_delta":
            latex_row = "Yang et al."
        elif tool_name == "DL4PatchCorrectness":
            latex_row = "Tian et al."
        else:
            latex_row = tool_name
        
        # Add values for each metric
        for i in range(1, len(row)):
            latex_row += f" & {row[i]}"
        
        latex_code.append(latex_row + " \\\\")
    
    # End the table
    if use_longtable:
        latex_code.extend([
            "\\end{longtable}",
            "\\end{center}"
        ])
    else:
        latex_code.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}"
        ])
    
    return "\n".join(latex_code)

def main():
    parser = argparse.ArgumentParser(description='Generate output from experiment results for a single metric.')
    parser.add_argument('experiment_type', help='Type of experiment to analyze')
    parser.add_argument('--aggregate', choices=['approach', 'bug', 'project'], default='approach',
                        help='How to aggregate the results (by approach, bug, or project)')
    parser.add_argument('--metric', 
                        choices=['Accuracy', 'Balanced Accuracy', 'Precision', 'Positive Recall', 'Negative Recall', 'F1 Score', 'MCC', 'Smooth MCC', 'Epsilon MCC', 'RS'],
                        default='Accuracy',
                        help='Metric to calculate (only one at a time)')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--include-wbc', action='store_true',
                        help='Include WBC baseline that predicts with fixed probabilities')
    parser.add_argument('--wbc-p-overfit', type=float, default=0.65,
                        help='Probability that WBC predicts overfitting (default: 0.65)')
    parser.add_argument('--confidence', type=float, default=80.0,
                        help='Confidence percentage for RS metric (default: 80.0)')
    parser.add_argument('--format', choices=['csv', 'latex'], default='latex',
                        help='Output format (csv or latex)')
    parser.add_argument('--longtable', action='store_true',
                        help='Use longtable environment for LaTeX output (for tables that span multiple pages)')
    parser.add_argument('--overall', action='store_true',
                        help='Generate an overall comparison table with all metrics for each tool')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Add cluster-aware bootstrap CIs next to point estimates (only with --overall)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    if args.overall:
        # Generate overall comparison table without grouping
        csv_rows = generate_overall_output(args.experiment_type, 
                                           args.include_wbc, args.wbc_p_overfit,
                                           args.confidence, args.bootstrap)
        
        if args.format == 'csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            print(f"Overall CSV comparison saved to {output_path}")
        else:  # args.format == 'latex'
            # Note that we're passing aggregate_by here only for the table title and label,
            # the metrics calculation ignores this when --overall is set
            latex_code = generate_overall_latex(args.experiment_type, "none", 
                                              csv_rows, use_longtable=args.longtable)
            with open(output_path, 'w') as f:
                f.write(latex_code)
            print(f"Overall LaTeX comparison saved to {output_path}")
    else:
        # Original single metric behavior
        csv_rows = generate_csv_output(args.experiment_type, args.aggregate, args.metric, 
                                     args.include_wbc, args.wbc_p_overfit, args.confidence)
        
        if args.format == 'csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            print(f"CSV file saved to {output_path}")
        else:  # args.format == 'latex'
            latex_code = generate_latex_output(args.experiment_type, args.aggregate, args.metric, 
                                           csv_rows, use_longtable=args.longtable)
            with open(output_path, 'w') as f:
                f.write(latex_code)
            print(f"LaTeX table saved to {output_path}")

if __name__ == "__main__":
    main()
