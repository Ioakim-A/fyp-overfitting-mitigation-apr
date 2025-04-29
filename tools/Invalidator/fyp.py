import subprocess
import os
import re
import javalang
import tempfile
import glob
import time
import shutil
import json

# Replace with your actual Java 11 installation path
JAVA_11_PATH = "/usr/lib/jvm/java-1.11.0-openjdk-amd64"

# Replace with the actual absolute path to the 8h deduplicated patch set
PATCHES_PATH = "/home/user/fyp-overfitting-mitigation-apr/patches_by_time/patches_8h_deduplicated"

def to_package_pattern(fully_qualified):
    base = fully_qualified.split('$', 1)[0]          # drop inner‑class part
    parts = base.split('.')
    if len(parts) > 1:
        parent_pkg = '.'.join(parts[:-1])            # strip last segment
    else:
        parent_pkg = base                            # defensive fallback
    return parent_pkg.replace('.', r'\.') + '.*'

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

def run_with_timeout(cmd, cwd, timeout):
    try:
        return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f"Command timed out: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, 124, stdout=e.stdout or '', stderr=e.stderr or '')

def find_classpath_dirs(base_dir):
    class_dirs = []
    patterns = [
        "**/build",                    
        "**/build-tests",              
        "**/classes",             
        "**/tests", 
        "**/test",              
        "**/test-classes",

    ]
    for pattern in patterns:
        class_dirs.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    return class_dirs

def replace_class_name_everywhere(code, original_class, new_class):
    return re.sub(rf'\b{original_class}\b', new_class, code)

def get_class_declaration_line(content, original_class, new_class):
    pattern = rf'(public\s+)?(abstract\s+)?class\s+{original_class}\b.*?\{{'
    match = re.search(pattern, content, flags=re.DOTALL)

    if not match:
        print(f"Could not extract full class declaration for {original_class}")
        return f"public class {new_class} extends junit.framework.TestCase {{"

    full_decl = match.group(0)
    return full_decl.replace(f'class {original_class}', f'class {new_class}')

def find_test_class_path(test_class, project_dir):
    relative_path = test_class.replace('.', '/') + '.java'
    matches = glob.glob(os.path.join(project_dir, '**', relative_path), recursive=True)
    if not matches:
        print(f"Could not find file for test class: {test_class}")
        return None
    return matches[0]

def extract_class_body_elements(content, original_class, new_class):
    method_map = {}
    field_declarations = set()
    inner_class_declarations = []
    constructor_declarations = []
    lines = content.splitlines()
    tree = javalang.parse.parse(content)

    def clean_code_block(block):
        open_comments = block.count("/*")
        close_comments = block.count("*/")
        if open_comments > close_comments:
            block += "\n*/"
        return block

    for _, class_decl in tree.filter(javalang.tree.ClassDeclaration):
        if class_decl.name != original_class:
            continue
        for decl in class_decl.body:
            if not decl.position:
                continue
            start_line = decl.position.line - 1
            end_line = start_line
            brace_count = 0
            found_open = False

            if isinstance(decl, javalang.tree.FieldDeclaration):
                field_lines = []
                while end_line < len(lines):
                    field_lines.append(lines[end_line])
                    # End of a field is usually marked with semicolon unless it's an initializer block
                    if ';' in lines[end_line] and lines[end_line].strip().endswith(';'):
                        break
                    end_line += 1

                field_code = clean_code_block("\n".join(field_lines))
                field_declarations.add(field_code)

            elif isinstance(decl, javalang.tree.MethodDeclaration):
                while end_line < len(lines):
                    brace_count += lines[end_line].count('{')
                    brace_count -= lines[end_line].count('}')
                    if '{' in lines[end_line]:
                        found_open = True
                    if found_open and brace_count == 0:
                        break
                    end_line += 1
                method_code = clean_code_block("\n".join(lines[start_line:end_line + 1]))
                if decl.name not in method_map:
                    method_map[decl.name] = []

                method_map[decl.name].append({
                    "code": method_code,
                    "annotations": decl.annotations,
                    "is_test": (
                                    any(getattr(ann, "name", "").endswith("Test") for ann in (decl.annotations or [])) or
                                    (decl.name.startswith("test") and 'assert' in method_code)
                                )
                })

            elif isinstance(decl, javalang.tree.ConstructorDeclaration):
                if decl.name != original_class:
                    continue
                while end_line < len(lines):
                    brace_count += lines[end_line].count('{')
                    brace_count -= lines[end_line].count('}')
                    if '{' in lines[end_line]:
                        found_open = True
                    if found_open and brace_count == 0:
                        break
                    end_line += 1
                constructor_code = "\n".join(lines[start_line:end_line + 1])
                constructor_code = re.sub(rf'\b{original_class}\b', new_class, constructor_code)
                constructor_code = clean_code_block(constructor_code)
                constructor_declarations.append(constructor_code)

            elif isinstance(decl, javalang.tree.ClassDeclaration):
                while end_line < len(lines):
                    brace_count += lines[end_line].count('{')
                    brace_count -= lines[end_line].count('}')
                    if '{' in lines[end_line]:
                        found_open = True
                    if found_open and brace_count == 0:
                        break
                    end_line += 1
                inner_class_code = clean_code_block("\n".join(lines[start_line:end_line + 1]))
                inner_class_declarations.append(inner_class_code)

            elif isinstance(decl, javalang.tree.InterfaceDeclaration):
                # Extract inner interfaces
                while end_line < len(lines):
                    brace_count += lines[end_line].count('{')
                    brace_count -= lines[end_line].count('}')
                    if '{' in lines[end_line]:
                        found_open = True
                    if found_open and brace_count == 0:
                        break
                    end_line += 1
                interface_code = clean_code_block("\n".join(lines[start_line:end_line + 1]))
                inner_class_declarations.append(interface_code)

    return method_map, field_declarations, inner_class_declarations, constructor_declarations


def extract_test_methods(test_class, class_path):
    try:
        with open(class_path, 'r') as f:
            content = f.read()

        tree = javalang.parse.parse(content)
        package = tree.package.name if tree.package else ''
        imports = [
            f"import {'static ' if imp.static else ''}{imp.path}{'.*' if imp.wildcard else ''};"
            for imp in tree.imports
        ]
        class_name = test_class.split('.')[-1]
        method_map, field_declarations, inner_classes, constructor_declarations = extract_class_body_elements(content, class_name, class_name)

        test_methods = [
            (name, i) for name, versions in method_map.items()
            for i, info in enumerate(versions) if info["is_test"]
        ]

        helper_methods = {
            (name, i): info["code"]
            for name, versions in method_map.items()
            for i, info in enumerate(versions)
            if not info["is_test"]
        }

        return {
            "package": package,
            "imports": imports,
            "class_name": class_name,
            "test_methods": test_methods,
            "helper_methods": helper_methods,
            "all_methods": method_map,
            "class_path": class_path,
            "fields": field_declarations,
            "inner_classes": inner_classes,
            "constructors": constructor_declarations,
            "source": content
        }

    except Exception as e:
        print(f"Error reading test class {test_class}: {str(e)}")
        return None

def create_test_class(class_info, methods, suffix, project_dir):
    package = class_info["package"]
    imports = class_info["imports"]
    class_name = f"{class_info['class_name']}{suffix}"
    original_class = class_info['class_name']

    content = [f"package {package};", ""]
    content.extend(imports)
    content.append("")
    class_declaration = get_class_declaration_line(class_info["source"], original_class, class_name)
    content.append(class_declaration)

    for field in sorted(class_info.get("fields", [])):
        content.append("")
        content.append(field)

    for constructor in class_info.get("constructors", []):
        renamed_constructor = replace_class_name_everywhere(constructor, original_class, class_name)
        content.append("")
        content.append(renamed_constructor)
    
    for method_key in methods:
        method_info = class_info["all_methods"][method_key[0]][method_key[1]]
        method_code = method_info["code"]
        # Rewrite new OriginalClass() → new NewClass()
        #method_code = method_code.replace(f"{original_class}", f"{class_name}")
        method_code = replace_class_name_everywhere(method_code, original_class, class_name)
        content.append("")
        content.append(method_code)

    for method_key, method_code in class_info["helper_methods"].items():
        if method_key in methods:
            continue
        method_code = replace_class_name_everywhere(method_code, original_class, class_name)
        content.append("")
        content.append(method_code)

    for inner_class in class_info.get("inner_classes", []):
        content.append("")
        content.append(inner_class)

    content.append("}")

    original_path = class_info["class_path"]
    output_dir = os.path.dirname(original_path)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{class_name}.java")
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))

    return f"{package}.{class_name}"

# Config
original_java_home = os.environ.get("JAVA_HOME")
original_path = os.environ.get("PATH")

os.environ["JAVA_HOME"] = JAVA_11_PATH
os.environ["PATH"] = f"{JAVA_11_PATH}/bin:{original_path}"

project_root = "../.."
invariants_dir = "fyp_data/processed_data/invariants"
log_path = "compile_failures.log"

# Add directories for patch data
patch_invariants_dir = os.path.join(invariants_dir, "patches")
patch_info_dir = "fyp_data/raw_data/patch_info"
patch_meta_dir = os.path.join(patch_info_dir, "fyp_patches")
overall_patches_file = f'{patch_info_dir}/patches.json'

# Ensure directories exist
os.makedirs(patch_invariants_dir, exist_ok=True)
os.makedirs(patch_meta_dir, exist_ok=True)

patch_dirs = {
    'Correct': f'{PATCHES_PATH}/correct',
    'Incorrect': f'{PATCHES_PATH}/overfitting'
}

patches = {}

for correctness, dir_path in patch_dirs.items():
    for patch_file in glob.glob(os.path.join(dir_path, '*.diff')):
        filename = os.path.basename(patch_file)
        parts = filename.split('_')
        if len(parts) >= 3:
            project = parts[0]
            version = parts[1]
            tool = parts[2].split('.')[0]
            key = f"{project}_{version}"
            patch_info = {
                'file': patch_file,
                'tool': tool,
                'correctness': correctness
            }
            patches.setdefault(key, []).append(patch_info)

patch_counter = 1
patch_number_to_filename = {}
overall_patches = []

start_time = time.time()

# Process each project-version pair
for project in os.listdir(os.path.join(project_root, "buggy")):
    for version_id in os.listdir(os.path.join(project_root, "buggy", project)):
        if version_id == '33_backup':
            continue
        print(f"\n==== Processing project: {project}, version: {version_id} ====")
        
        # Check if we have patches for this project-version
        project_version_key = f"{project}_{version_id}"
        project_patches = patches.get(project_version_key, [])
        
        # Collect info from defects4j
        result = subprocess.run(
            ['defects4j', 'info', '-p', project, '-b', version_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        output = result.stdout
        failing_tests = re.findall(r'- ([\w\.]+)::([\w\d_]+)', output)
        modified_classes = re.findall(r'- ([\w\.]+)', output.split("List of modified sources:")[-1])
        ppt_select_pattern = '|'.join([cls.replace('.', r'\.') + '.*' for cls in modified_classes])
        if project == 'Math' or project == 'Closure' or project == 'Mockito':
            ppt_select_pattern = '.*'

        # Copy the buggy and fixed project directories
        buggy_src = os.path.join(project_root, "buggy", project, version_id)
        fixed_src = os.path.join(project_root, "fixed", project, version_id)
        temp_dir = tempfile.mkdtemp(prefix=f"{project}_{version_id}_")
        compilation_failed = False
        buggy_copy = os.path.join(temp_dir, "buggy")
        fixed_copy = os.path.join(temp_dir, "fixed")
        shutil.copytree(buggy_src, buggy_copy)
        shutil.copytree(fixed_src, fixed_copy)

        # Generate new test classes only once
        processed_classes = set()
        generated_classes = []  # Store (test_class, failing_class_name, passing_class_name)

        for test_class, test_method in failing_tests:
            if test_class in processed_classes:
                continue
            processed_classes.add(test_class)

            original_path = find_test_class_path(test_class, buggy_copy)
            if not original_path:
                continue
            class_info = extract_test_methods(test_class, original_path)
            if not class_info:
                continue

            failing_methods = [
                key for key in class_info["test_methods"] if key[0] in [m for cls, m in failing_tests if cls == test_class]
            ]
            passing_methods = [
                key for key in class_info["test_methods"] if key not in failing_methods
            ]

            # Create test classes in buggy copy
            failing_class = create_test_class(class_info, failing_methods, "FailingTests", buggy_copy)
            passing_class = create_test_class(class_info, passing_methods, "PassingTests", buggy_copy)

            # Also copy the new files to the fixed copy
            src_dir = os.path.dirname(class_info['class_path'])
            dest_dir = src_dir.replace(buggy_copy, fixed_copy)
            os.makedirs(dest_dir, exist_ok=True)
            for suffix in ["FailingTests", "PassingTests"]:
                base_name = class_info['class_name'] + suffix + ".java"
                shutil.copy(os.path.join(src_dir, base_name), os.path.join(dest_dir, base_name))

            generated_classes.append((test_class, failing_class, passing_class))

        # For each copy (buggy and fixed), compile and run Daikon
        for version_label, root_dir in [("b", buggy_copy), ("f", fixed_copy)]:

            print(f"\nCompiling project {version_label} ({root_dir})...")
            compile_result = subprocess.run(
                ['defects4j', 'compile'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=root_dir
            )

            fallback_used = False

            if compile_result.returncode != 0:
                print("Compilation failed. Attempting fallback...")

                # Fallback: delete generated test classes (*.FailingTests.java, *.PassingTests.java)
                for pattern in ["*FailingTests.java", "*PassingTests.java"]:
                    for test_file in glob.glob(os.path.join(root_dir, "**", pattern), recursive=True):
                        print(f"Deleting {test_file}")
                        try:
                            os.remove(test_file)
                        except Exception as e:
                            print(f"Error deleting {test_file}: {e}")

                # Try compiling again after cleanup
                compile_result = subprocess.run(
                    ['defects4j', 'compile'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=root_dir
                )

                if compile_result.returncode != 0:
                    print("Fallback compilation also failed.")
                    with open(log_path, "a") as log_file:
                        log_file.write("=== Fallback Compilation Failed ===\n")
                        log_file.write(f"Project: {project}, Version: {version_id}, Type: {version_label}\n")
                        log_file.write(f"Temp dir: {temp_dir}\n")
                        log_file.write("STDOUT:\n" + compile_result.stdout + "\n")
                        log_file.write("STDERR:\n" + compile_result.stderr + "\n")
                        log_file.write("===========================\n\n")
                    compilation_failed = True
                    continue
                else:
                    fallback_used = True

            out_dir = os.path.join(invariants_dir, version_label, project, version_id)
            os.makedirs(out_dir, exist_ok=True)

            class_dirs = find_classpath_dirs(root_dir)
            classpath = ":".join(
                [f'{os.environ.get("DAIKONDIR")}/daikon.jar'] +
                class_dirs +
                glob.glob(os.path.join(root_dir, "lib", "*.jar")) 
            )

            for test_class, failing_class, passing_class in generated_classes:
                base = test_class.split(".")[-1]
                trace_fail = f"{base}_{version_label}_failing.dtrace.gz"
                trace_pass = f"{base}_{version_label}_passing.dtrace.gz"

                failing_runner = test_class if fallback_used else failing_class
                passing_runner = test_class if fallback_used else passing_class

                # Run Chicory - Failing
                run_with_timeout([
                    'java', '-cp', classpath,
                    'daikon.Chicory',
                    f'--dtrace-file={trace_fail}',
                    f'--ppt-select-pattern={ppt_select_pattern}',
                    'org.junit.runner.JUnitCore', failing_runner
                ], cwd=root_dir, timeout=300)

                # Run Chicory - Passing
                run_with_timeout([
                    'java', '-cp', classpath,
                    'daikon.Chicory',
                    f'--dtrace-file={trace_pass}',
                    f'--ppt-select-pattern={ppt_select_pattern}',
                    'org.junit.runner.JUnitCore', passing_runner
                ], cwd=root_dir, timeout=300)

                # Analyze traces with Daikon and save results
                for trace_file, label in [(trace_fail, "failing"), (trace_pass, "passing")]:
                    proc = run_with_timeout([
                                'java', 'daikon.Daikon', trace_file
                            ], cwd=root_dir, timeout=300)
                    result_path = os.path.join(out_dir, f"result_{label}.txt")
                    with open(result_path, "a") as f:
                        f.write(proc.stdout)

        # Now handle patches for this project-version
        print(f"\n==== Processing {len(project_patches)} patches for {project}_{version_id} ====")
        for patch_info in project_patches:
            patch_file_path = patch_info['file']
            tool = patch_info['tool']
            correctness = patch_info['correctness']
            
            print(f"\nProcessing patch {patch_counter}: {os.path.basename(patch_file_path)}")
            
            # Create temp directory for this patch
            patch_temp_dir = tempfile.mkdtemp(prefix=f"{project}_{version_id}_patch{patch_counter}_")
            patched_copy = os.path.join(patch_temp_dir, "patched")
            
            # Copy buggy project as base for patch
            shutil.copytree(buggy_src, patched_copy)
            
            # Apply the patch
            print(f"Applying patch from {patch_file_path}")
            convert_line_endings_to_unix(patch_file_path)
            normalize_java_files(patched_copy)
            patch_result = subprocess.run(
                ["patch", "-p1", "--ignore-whitespace", "-i", patch_file_path, "-t"],
                cwd=patched_copy,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if patch_result.returncode != 0:
                print(f"Failed to apply patch: {patch_result.stderr}")
                shutil.rmtree(patch_temp_dir)
                continue

            if project == 'Math' and version_id == '33':
                compile_result = subprocess.run(
                ['defects4j', 'compile'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=patched_copy
                )
                if compile_result.returncode != 0:
                    shutil.rmtree(patched_copy)
                    shutil.copytree(os.path.join(project_root, "buggy", project, '33_backup'), patched_copy)
                    normalize_java_files(patched_copy)
                    patch_result = subprocess.run(
                        ["patch", "-p1", "--ignore-whitespace", "-i", patch_file_path, "-t"],
                        cwd=patched_copy,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                
            # Create test classes in patched copy
            for test_class, test_method in failing_tests:
                if test_class not in processed_classes:  # This should never happen but just to be safe
                    continue
                    
                original_path = find_test_class_path(test_class, patched_copy)
                if not original_path:
                    continue
                class_info = extract_test_methods(test_class, original_path)
                if not class_info:
                    continue

                failing_methods = [
                    key for key in class_info["test_methods"] if key[0] in [m for cls, m in failing_tests if cls == test_class]
                ]
                passing_methods = [
                    key for key in class_info["test_methods"] if key not in failing_methods
                ]

                # Create test classes in patched copy
                failing_class = create_test_class(class_info, failing_methods, "FailingTests", patched_copy)
                passing_class = create_test_class(class_info, passing_methods, "PassingTests", patched_copy)
            
            # Compile the patched project
            print(f"Compiling patched project...")
            compile_result = subprocess.run(
                ['defects4j', 'compile'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=patched_copy
            )
            
            patch_compilation_failed = False
            fallback_used = False
            
            if compile_result.returncode != 0:
                print("Compilation failed. Attempting fallback...")
                
                # Fallback: delete generated test classes
                for pattern in ["*FailingTests.java", "*PassingTests.java"]:
                    for test_file in glob.glob(os.path.join(patched_copy, "**", pattern), recursive=True):
                        print(f"Deleting {test_file}")
                        try:
                            os.remove(test_file)
                        except Exception as e:
                            print(f"Error deleting {test_file}: {e}")
                
                # Try compiling again after cleanup
                compile_result = subprocess.run(
                    ['defects4j', 'compile'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=patched_copy
                )
                
                if compile_result.returncode != 0:
                    print("Fallback compilation also failed.")
                    with open(log_path, "a") as log_file:
                        log_file.write("=== Fallback Compilation Failed for Patch ===\n")
                        log_file.write(f"Project: {project}, Version: {version_id}, Patch ID: {patch_counter}\n")
                        log_file.write(f"Temp dir: {patch_temp_dir}\n")
                        log_file.write("STDOUT:\n" + compile_result.stdout + "\n")
                        log_file.write("STDERR:\n" + compile_result.stderr + "\n")
                        log_file.write("===========================\n\n")
                    patch_compilation_failed = True
                else:
                    fallback_used = True
            
            if not patch_compilation_failed:
                # Create output directory for this patch
                out_dir = os.path.join(patch_invariants_dir, str(patch_counter))
                os.makedirs(out_dir, exist_ok=True)
                
                class_dirs = find_classpath_dirs(patched_copy)
                classpath = ":".join(
                    [f'{os.environ.get("DAIKONDIR")}/daikon.jar'] +
                    class_dirs +
                    glob.glob(os.path.join(patched_copy, "lib", "*.jar")) 
                )
                
                for test_class, failing_class, passing_class in generated_classes:
                    base = test_class.split(".")[-1]
                    trace_fail = f"{base}_p{patch_counter}_failing.dtrace.gz"
                    trace_pass = f"{base}_p{patch_counter}_passing.dtrace.gz"
                    
                    failing_runner = test_class if fallback_used else failing_class
                    passing_runner = test_class if fallback_used else passing_class
                    
                    # Run Chicory - Failing
                    run_with_timeout([
                        'java', '-cp', classpath,
                        'daikon.Chicory',
                        f'--dtrace-file={trace_fail}',
                        f'--ppt-select-pattern={ppt_select_pattern}',
                        'org.junit.runner.JUnitCore', failing_runner
                    ], cwd=patched_copy, timeout=300)
                    
                    # Run Chicory - Passing
                    run_with_timeout([
                        'java', '-cp', classpath,
                        'daikon.Chicory',
                        f'--dtrace-file={trace_pass}',
                        f'--ppt-select-pattern={ppt_select_pattern}',
                        'org.junit.runner.JUnitCore', passing_runner
                    ], cwd=patched_copy, timeout=300)
                    
                    # Analyze traces with Daikon and save results
                    for trace_file, label in [(trace_fail, "failing"), (trace_pass, "passing")]:
                        proc = run_with_timeout([
                            'java', 'daikon.Daikon', trace_file
                        ], cwd=patched_copy, timeout=300)
                        result_path = os.path.join(out_dir, f"results_{label}.txt")
                        with open(result_path, "a") as f:
                            f.write(proc.stdout)
                
                # Save patch metadata
                patch_metadata = {
                    "ID": f"Patch{patch_counter}",
                    "tool": tool,
                    "correctness": correctness,
                    "bug_id": version_id,
                    "project": project
                }
                
                with open(os.path.join(patch_meta_dir, f"Patch{patch_counter}.json"), 'w') as f:
                    json.dump(patch_metadata, f, indent=2)
                patch_number_to_filename[f"Patch{patch_counter}"] = os.path.basename(patch_file_path)

                
                # Add to overall patches list
                overall_patches.append({
                    "project": project,
                    "bug_id": version_id,
                    "patch_file": f"Patch{patch_counter}",
                    "id": patch_counter
                })
                
                # Update overall patches JSON
                with open(overall_patches_file, 'w') as f:
                    json.dump(overall_patches, f, indent=2)
                
                # Increment patch counter
                patch_counter += 1
            
            # Clean up temp directory for this patch
            shutil.rmtree(patch_temp_dir)
                
        if not compilation_failed:
            shutil.rmtree(temp_dir)
        else:
            print(f"Preserved temp directory due to compilation failure: {temp_dir}")

mapping_file = os.path.join(patch_info_dir, "patch_number_to_filename.json")
with open(mapping_file, "w") as f:
    json.dump(patch_number_to_filename, f, indent=2)
print(f"Wrote patch-number mapping to {mapping_file}")

end_time = time.time()
print(f"Took {end_time - start_time:.2f} seconds")

os.environ["JAVA_HOME"] = original_java_home
os.environ["PATH"] = original_path