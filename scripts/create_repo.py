import argparse
import os
from huggingface_hub import Repository, create_repo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create repo')
    parser.add_argument('-r', '--repo_name', required=True, type=str)
    parser.add_argument('-o', '--output_dir', required=True, type=str)
    arg = parser.parse_args()

    repo_id = create_repo(arg.repo_name, exist_ok=True, repo_type="dataset").repo_id
    repo = Repository(arg.output_dir, clone_from=repo_id, repo_type="dataset")

    with open(os.path.join(arg.output_dir, ".gitignore"), "w+") as gitignore:
        if "wandb" not in gitignore:
            gitignore.write("wandb\n")

    # Ensure large txt files can be pushed to the Hub with git-lfs
    with open(os.path.join(arg.output_dir, ".gitattributes"), "r+") as f:
        git_lfs_extensions = f.read()
        if "*.csv" not in git_lfs_extensions:
            f.write("*.csv filter=lfs diff=lfs merge=lfs -text")

