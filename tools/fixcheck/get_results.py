import os
import csv
import argparse

ALL_PATCHES_DIR = "../../all_patches"

def determine_actual_correctness(patch_name):
    """Determine actual correctness based on where the patch is located in FYP_DATASET"""
    correct_path = os.path.join(ALL_PATCHES_DIR, "correct", f'{patch_name}.diff')
    overfitting_path = os.path.join(ALL_PATCHES_DIR, "overfitting", f'{patch_name}.diff')
    
    if os.path.exists(correct_path):
        return "correct"
    elif os.path.exists(overfitting_path):
        return "overfitting"
    else:
        # This is never executed in the current setup, but we keep it for safety
        print(f"Warning: {patch_name} not found in either correct or overfitting directories.")
        return "unknown"

def determine_prediction(assertion_failing_prefixes):
    """Determine prediction based on the values in the report.csv"""
    failing = int(assertion_failing_prefixes) if assertion_failing_prefixes else 0
    
    if failing > 0: 
        return "overfitting"
    else: #could not generate a failing assertion
        return "correct"

def process_directory(directory, method):
    """Process all subdirectories and collect results"""
    results = []
    
    # Walk through all subdirectories in the given directory
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
        
        # Check if the method directory exists and contains report.csv
        method_path = os.path.join(subdir_path, method)
        report_path = os.path.join(method_path, "report.csv")

        # For every compiled patch, check if fixcheck was able to generate a failing assertion which marks a patch as overfitting
        prediction = 'correct'

        correctness = determine_actual_correctness(subdir)
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    # Get the first row
                    for row in reader:
                        assertion_failing_prefixes = row.get('assertion_failing_prefixes', '0')
                        
                        # Determine prediction
                        prediction = determine_prediction(assertion_failing_prefixes)
                                   
                        # We only need the first row
                        break
            except Exception as e:
                print(f"Error processing {report_path}: {e}")

        results.append({
                        'patch_name': f'{subdir}.diff',
                        'correctness': correctness,
                        'prediction': prediction
                        })
         
    return results

def main():
    results = process_directory('fixcheck-output/defects-repairing', 'replit-code-llm')
    output_path = 'fixcheck_results_8h_test.csv'
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['patch_name', 'correctness', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    main()
