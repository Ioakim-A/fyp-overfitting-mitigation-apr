import os
import shutil

context_size = "max"  # context size of the patch, options "3" or "max"

# filter by which patches can be handled by current project
project_versions = {
    "Chart": ["12", "25", "5", "17", "8", "20", "24", "4", "15", "7", "11", "13", "18", "1", "9", "26",],
    "Lang": ["6", "44", "57", "59", "33", "46", "58", "43", "51", "10", "63", "55", "45",],
    "Closure": ["129", "22", "14", "57", "46", "86", "21", "115", "73", "2", "126", "13", "19", "62", "10", "48",],
    "Math": ["84", "70", "95", "6", "5", "8", "41", "88", "20", "69", "57", "50", "65", "32", "28", "15", "59", "33", "82", "42", "7", "58", "11", "73", "105", "2", "81", "49", "89", "71", "62", "79", "101", "85", "64", "96", "63", "30", "53", "80", "75", "97", "34",],  # "33_backup removed"
    "Mockito": ["29", "38",],
    "Time": ["7", "11"],
}

# get directory of repairllama patches
current_directory = os.getcwd()
print(f"Current directory: {current_directory}")
repairllama_patches_dir = os.path.join(current_directory, "patches_repairllama", "diff_file",
                                       f"context_size_{context_size}", "evaluation_defects4j_repairllama_ir4_or2_martin",)
print(f"RepairLlama patches directory: {repairllama_patches_dir}")

# get directory of filtered patches
output_base_dir = os.path.join(
    current_directory, "patches_repairllama", "filtered", f"context_size_{context_size}_filtered",)

# Remove the output_base_dir if it exists
if os.path.exists(output_base_dir):
    shutil.rmtree(output_base_dir)


os.makedirs(output_base_dir, exist_ok=True)

valid_correct_patches = []
valid_overfitting_patches = []
for patch_type in ["correct", "overfitting"]:
    patch_dir = os.path.join(repairllama_patches_dir, patch_type)
    if not os.path.exists(patch_dir):
        print(f"Patch directory does not exist: {patch_dir}")
        continue

    # get all files in the patch directory
    files = os.listdir(patch_dir)
    print(f"Files in {patch_type} directory: {files}")
    print(f"Number of files in {patch_type} directory: {len(files)}\n\n\n")

    # filter files by project and version
    for file in files:
        for project, versions in project_versions.items():
            if any(f"{project}_{version}" in file for version in versions):
                if patch_type == "correct":
                    valid_correct_patches.append(file)
                else:
                    valid_overfitting_patches.append(file)
                break

print("Valid Correct Patches:")
print(valid_correct_patches)
print("Number of valid correct patches:", len(valid_correct_patches))
print("\n\n\n")
print("Valid Overfitting Patches:")
print(valid_overfitting_patches)
print("Number of valid overfitting patches:", len(valid_overfitting_patches))

# copy filtered patches to the output directory
for patch_type, valid_patches in zip(["correct", "overfitting"], [valid_correct_patches, valid_overfitting_patches]):
    output_dir = os.path.join(output_base_dir, patch_type)
    os.makedirs(output_dir, exist_ok=True)
    for file in valid_patches:
        src_file = os.path.join(repairllama_patches_dir, patch_type, file)
        dst_file = os.path.join(output_dir, file)
        print(f"Copying {src_file} to {dst_file}")
        shutil.copy(src_file, dst_file)


# # filter patches
# for patch_type in ["correct", "overfitting"]:
#     output_dir = os.path.join(output_base_dir, patch_type)
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Output directory for {patch_type}: {output_dir}")
