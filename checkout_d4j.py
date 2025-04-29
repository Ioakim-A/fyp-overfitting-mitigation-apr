import subprocess
from pathlib import Path
import sys
import os

JAVA_11_PATH = "/usr/lib/jvm/java-1.11.0-openjdk-amd64"

PATCH_ROOT = Path("all_patches")
BUGGY_ROOT = Path("buggy")
FIXED_ROOT = Path("fixed")

def find_project_bug_pairs(patch_root: Path):
    """Scan the 'correct' and 'overfitting' subdirs for .diff files,
    extract (project, bug) pairs, and dedupe them."""
    pairs = set()
    for sub in ("correct", "overfitting"):
        base_dir = patch_root / sub
        if not base_dir.is_dir():
            print(f"Warning: {base_dir} does not exist or is not a directory", file=sys.stderr)
            continue
        for diff_path in base_dir.rglob("*.diff"):
            stem = diff_path.stem
            parts = stem.split("_", 2)
            if len(parts) < 2:
                print(f"Warning: filename doesn't match pattern, skipping: {diff_path}", file=sys.stderr)
                continue
            project, bug = parts[0], parts[1]
            pairs.add((project, bug))
    return sorted(pairs)

def checkout_versions(pairs, buggy_root: Path, fixed_root: Path):
    """For each (project, bug), create directories and run defects4j checkout."""
    for project, bug in pairs:
        if project == 'Time':
            continue
        print(f"\n=== Processing {project} bug {bug} ===")
        buggy_dir = buggy_root / project / bug
        fixed_dir = fixed_root / project / bug

        buggy_dir.mkdir(parents=True, exist_ok=True)
        fixed_dir.mkdir(parents=True, exist_ok=True)

        # buggy version: "-v <bug>b"
        print(f"-> Checking out BUGGY version into {buggy_dir}")
        subprocess.run([
            "defects4j", "checkout",
            "-p", project,
            "-v", f"{bug}b",
            "-w", str(buggy_dir)
        ], check=True)

        # fixed version: "-v <bug>f"
        print(f"-> Checking out FIXED version into {fixed_dir}")
        subprocess.run([
            "defects4j", "checkout",
            "-p", project,
            "-v", f"{bug}f",
            "-w", str(fixed_dir)
        ], check=True)

def main():
    original_java_home = os.environ.get("JAVA_HOME")
    original_path = os.environ.get("PATH")
    os.environ["JAVA_HOME"] = JAVA_11_PATH
    os.environ["PATH"] = f"{JAVA_11_PATH}/bin:{original_path}"

    pairs = find_project_bug_pairs(PATCH_ROOT)
    if not pairs:
        print("No valid patch files found under 'all_patches'. Exiting.", file=sys.stderr)
        sys.exit(1)

    checkout_versions(pairs, BUGGY_ROOT, FIXED_ROOT)

    os.environ["JAVA_HOME"] = original_java_home
    os.environ["PATH"] = original_path

if __name__ == "__main__":
    main()