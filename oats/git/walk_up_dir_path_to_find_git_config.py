#!/usr/bin/env python3

"""Walk up the directory tree to find the first directory containing .git/config."""

import argparse
import logging
import os
import traceback
from typing import Tuple
from oats.log import cl

log = cl("find_git_config")

def walk_up_dir_path_to_find_git_config(dir_path: str) -> Tuple[bool, str]:
    """Walk up from *dir_path* and return the first parent directory that
    contains a ``.git/config`` file.

    Returns:
        ``(True, repo_dir)`` on success, ``(False, "")`` on failure.
    """
    try:
        current = os.path.abspath(dir_path)

        while True:
            git_config = os.path.join(current, ".git", "config")
            if os.path.isfile(git_config):
                return True, current

            parent = os.path.dirname(current)
            if parent == current:  # reached the filesystem root
                return False, ""

            current = parent

    except Exception:
        log.error(
            f"### Sorry!! git/walk_up_dir_path_to_find_git_config.py "
            f"Failed to find git repo config dir: {dir_path} with error:\n"
            f"```\n{traceback.format_exc()}\n```\n"
        )
        return False, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk up the directory tree to find the git repo root.")
    parser.add_argument("-d", "--target_dir", default=".", help="Target directory to start walking up from (default: .)")
    args = parser.parse_args()
    success, repo_dir = walk_up_dir_path_to_find_git_config(args.target_dir)
    if success:
        print(repo_dir)
    else:
        print("", end="")


if __name__ == "__main__":
    main()
