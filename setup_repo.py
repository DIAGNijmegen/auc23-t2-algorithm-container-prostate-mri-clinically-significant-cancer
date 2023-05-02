import argparse
import json
import os
import subprocess
import shutil

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trained-models-path", required=True)
parser.add_argument("--task-name", required=True)
parser.add_argument("--input_interfaces", nargs="+", required=True)
parser.add_argument("--output_interfaces", nargs="+", required=True)
parser.add_argument("--roi_segmentation_interface", required=True)
parser.add_argument("--new-repo-path", required=True)
args = parser.parse_args()
args.new_repo_path = os.path.expanduser(args.new_repo_path)

print("Copying the repo")
shutil.copytree(".", args.new_repo_path, ignore=shutil.ignore_patterns('.git'))

# Set roi_segmentation_interface to null if it is "none"
roi_segmentation_interface = None if args.roi_segmentation_interface.lower() == "none" else args.roi_segmentation_interface

# Create task_config.json
task_config = {
    "input_interfaces": args.input_interfaces,
    "output_interfaces": args.output_interfaces,
    "roi_segmentation_interface": roi_segmentation_interface
}

with open(os.path.join(args.new_repo_path, "task_config.json"), "w") as f:
    json.dump(task_config, f, indent=2)

# Create artifacts directory and copy files
src_base = os.path.join(args.trained_models_path, "nnUNet", "3d_fullres", args.task_name, "ClassifierTrainer__UniversalClassifierPlansv1.0")
dst_base = os.path.join(args.new_repo_path, "artifacts", "nnUNet", "3d_fullres", args.task_name, "ClassifierTrainer__UniversalClassifierPlansv1.0")

files_to_copy = [
    (os.path.join(src_base, "plans.pkl"), os.path.join(dst_base, "plans.pkl")),
    (os.path.join(src_base, "all", "model_final_checkpoint.model"), os.path.join(dst_base, "all", "model_final_checkpoint.model")),
    (os.path.join(src_base, "all", "model_final_checkpoint.model.pkl"), os.path.join(dst_base, "all", "model_final_checkpoint.model.pkl")),
]

print("copying files:")
for src, dst in files_to_copy:
    print(f"{dst}...")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
