"""
Adapted from: run-fixcheck-defect-repairing.py
"""

import os
import sys
import subprocess
import pandas as pd
import shutil
import re


# Replace with the actual path to your Java 11 installation
JAVA_11_PATH = '/usr/lib/jvm/java-1.11.0-openjdk-amd64'

# Replace with the actual absolute path to fixcheck
os.environ['FIXCHECK'] = '/home/user/workspace/FYP/Tools/fixcheck'
FIXCHECK = os.getenv('FIXCHECK')

# Replace with the actual absolute path to fixcheck/tmp
os.environ['DEFECT_REPAIRING_DATASET'] = '/home/user/workspace/FYP/Tools/fixcheck/tmp'
DEFECT_REPAIRING_DATASET = os.getenv('DEFECT_REPAIRING_DATASET')

# Replace with the actual absolute path to the replication package
BASE_BUG_PATCH_DIR = '/home/user/x/fyp-overfitting-mitigation-apr'

def log_failed_patch_compilation(patch_name, return_code, std_out, std_err):
    with open('failed_compilations.log', "a") as log_file:
        log_file.write(f"--- Failed to compile patch: {patch_name} ---\n")
        log_file.write(f"Return Code: {return_code}\n")
        log_file.write(f"Stdout:\n{std_out}\n")
        log_file.write(f"Stderr:\n{std_err}\n")
        log_file.write(f"{'-'*60}\n")

def convert_line_endings_to_unix(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    content = content.replace(b'\r\n', b'\n')  # Windows CRLF -> Unix LF
    with open(file_path, 'wb') as f:
        f.write(content)

def normalize_java_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java"):
                convert_line_endings_to_unix(os.path.join(root, file))

original_java_home = os.environ.get("JAVA_HOME")
original_path = os.environ.get("PATH")
os.environ["JAVA_HOME"] = JAVA_11_PATH
os.environ["PATH"] = f"{JAVA_11_PATH}/bin:{original_path}"

dataset_csv = 'experiments/8h-deduplicated_processed.csv'
outputs_dir = 'fixcheck-output'

assertion_generation = 'replit-code-llm'

df = pd.read_csv(dataset_csv)

for index, row in df.iterrows():
    subject_id = row['id']
    if os.path.isdir(f'{outputs_dir}/defects-repairing/{subject_id}'):
        continue
    patch_name = f'{subject_id}.diff'
    classification = 'correct' if row['correctness'] == 'Correct' else 'overfitting'
    base_dir = row['base_dir']
    patch_file_location = f'{BASE_BUG_PATCH_DIR}/all_patches/{classification}/{patch_name}'
    buggy_project_location = f'{BASE_BUG_PATCH_DIR}/buggy/{base_dir}'

    patch_dest_dir = os.path.join(DEFECT_REPAIRING_DATASET, base_dir)

    if os.path.exists(patch_dest_dir):
        shutil.rmtree(patch_dest_dir)

    shutil.copytree(buggy_project_location, patch_dest_dir)
    convert_line_endings_to_unix(patch_file_location)
    normalize_java_files(patch_dest_dir)
    try:
        result = subprocess.run(
            ["patch", "-p1", "--ignore-whitespace", "-i", patch_file_location, "-t"],
            cwd=patch_dest_dir,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch {patch_name} with standard patch command")
        exit()
    compiled = False
    try:
        compile_result = subprocess.run(
            ["defects4j", "compile"],
            cwd=patch_dest_dir,
            capture_output=True,
            text=True,
            check=True
        )
        compiled = True
        print(f"Successfully compiled patch {patch_name}")
    except subprocess.CalledProcessError as e:
        if os.path.exists(patch_dest_dir):
            shutil.rmtree(patch_dest_dir)

        if base_dir == 'Math/33':
            buggy_project_location = f'{BASE_BUG_PATCH_DIR}/buggy/Math/33_backup'
            patch_dest_dir = os.path.join(DEFECT_REPAIRING_DATASET, base_dir)

            if os.path.exists(patch_dest_dir):
                shutil.rmtree(patch_dest_dir)

            shutil.copytree(buggy_project_location, patch_dest_dir)
            normalize_java_files(patch_dest_dir)
            try:
                compile_result = subprocess.run(
                    ["defects4j", "compile"],
                    cwd=patch_dest_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                compiled = True
                print(f"Successfully compiled patch {patch_name}")
            except subprocess.CalledProcessError as e:
                log_failed_patch_compilation(patch_name, e.returncode, e.stdout, e.stderr)
        else:
            log_failed_patch_compilation(patch_name, e.returncode, e.stdout, e.stderr)

    if not compiled:
        continue

    print(f'Running FixCheck for subject: {subject_id}')
    print(f'assertion generation: {assertion_generation}')
    subject_data = df[df['id'] == subject_id]

    # Get and setup the subject data
    project = subject_data['project'].values[0]
    bug = subject_data['bug'].values[0]
    #subject_base_dir = os.path.join('/tmp', subject_data['base_dir'].values[0])

    #patch_base_dir = project+str(bug)+"b"
    #subject_base_dir = os.path.join(DEFECT_REPAIRING_DATASET, f'tmp/{subject_id}/{patch_base_dir}')
    
    subject_base_dir = os.path.join(DEFECT_REPAIRING_DATASET, base_dir)

    def build_classpath(subject_base_dir,main_dep,test_classes_path):
        # Split main dep in char ':' and join each part with subject_base_dir
        subject_cp = ':'.join([os.path.join(subject_base_dir, dep) for dep in main_dep.split(':')])
        subject_cp = subject_cp+':'+test_classes_path
        return subject_cp


    # Dependencies
    main_dep = subject_data['main_dep'].values[0]
    test_classes = subject_data['tests_build'].values[0]
    test_classes_path = os.path.join(subject_base_dir, test_classes)
    subject_cp = build_classpath(subject_base_dir,main_dep,test_classes_path)
    # Classes and methods
    target_test = subject_data['target_test'].values[0]
    target_test_methods = subject_data['target_test_methods'].values[0]
    tests_src_dir = subject_data['tests_src_dir'].values[0]
    target_test_dir = os.path.join(subject_base_dir, tests_src_dir)
    target_class = subject_data['target_class'].values[0]
    input_class = subject_data['input_class'].values[0]
    failure_log = os.path.join(subject_base_dir, 'failing_tests')

    # Run FixCheck
    subprocess.run(f'./fixcheck.sh {subject_cp} {test_classes_path} {target_test} {target_test_methods} {target_test_dir} {target_class} {input_class} {failure_log} {assertion_generation}', shell=True)

    # Move all outputs to a folder specific to the current subject
    output_file = os.path.join(outputs_dir, subject_id+'-report.csv')
    subject_output_folder = os.path.join(outputs_dir, f'defects-repairing/{subject_id}/{assertion_generation}')
    print(f'Moving all outputs to dir: {subject_output_folder}')
    if not os.path.exists(subject_output_folder):
        os.makedirs(subject_output_folder)
    subprocess.run(f'mv {outputs_dir}/report.csv {subject_output_folder}', shell=True)
    subprocess.run(f'mv {outputs_dir}/scores-failing-tests.csv {subject_output_folder}', shell=True)
    subprocess.run(f'mv {outputs_dir}/failing-tests {subject_output_folder}', shell=True)
    subprocess.run(f'mv {outputs_dir}/passing-tests {subject_output_folder}', shell=True)
    subprocess.run(f'mv {outputs_dir}/non-compiling-tests {subject_output_folder}', shell=True)
    subprocess.run(f'mv log.out {subject_output_folder}', shell=True)

os.environ["JAVA_HOME"] = original_java_home
os.environ["PATH"] = original_path  