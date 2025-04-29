"""
Adapted from original mipi_sample.py
"""

import json
import csv
import time
import glob
import os

from mipi.base_codemeaning_predictor import PatchInfo
from mipi.mipi_app import Mipi
import argparse


def load_patches_from_file(patches_json_file):
    with open(patches_json_file, "r", encoding="utf8") as file:
        patches_json = json.load(file)
    patches = []
    for jo_patch in patches_json:
        patch = PatchInfo()
        patch.from_json(jo_patch)
        patches.append(patch)
    return patches


if __name__ == '__main__':
 
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Run MIPI evaluation on patches')
    parser.add_argument('dataset_json_file', 
                        help='Path to the dataset JSON file')
    args = parser.parse_args()
    dataset_json_file = args.dataset_json_file
    patches = load_patches_from_file(dataset_json_file)

    obj_mipi = Mipi()

    correctness = {os.path.basename(f): os.path.basename(os.path.dirname(f))
                for f in glob.glob('../../../../patches_by_time/patches_8h_deduplicated/*/*.diff')}

    results_file = f"{dataset_json_file.split('/')[-1].split('.')[0]}_results.csv"
    start_time = time.time()
    with open(results_file, mode='w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['patch_name','correctness','prediction'])
        for p in patches:
            patch_id = p.patch_id
            correct_prediction_made = False
            try:
                rs = obj_mipi.evaluate(p)
                prediction = 'overfitting' if rs.predicted == 'incorrect' else 'correct'
            except:
                continue
            writer.writerow([patch_id, correctness[patch_id], prediction])
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print results
    print(f"Prediction completed in {elapsed_time:.2f} seconds")


