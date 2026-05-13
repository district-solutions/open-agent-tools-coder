#!/usr/bin/env python3
"""
Git Commit to Pandas DataFrame Converter

Extracts Git commit history and converts it into a structured pandas DataFrame,
sorted by commit date in descending order. Can save the result as a Parquet file.
"""

import argparse
import os
import sys
import pandas as pd
from git import Repo
from oats.log import cl

log = cl("git_to_df")


def extract_git_commits(repo_path: str) -> pd.DataFrame:
    """Extract Git commit history from a repository and return as a pandas DataFrame.

    Iterates over all commits in the repository, collecting SHA, author, date,
    message, and parent hashes. Results are sorted by commit date in descending order.

    Args:
        repo_path: Path to the Git repository.

    Returns:
        DataFrame with columns: ``id``, ``author_name``, ``author_email``,
        ``commit_date``, ``message``, ``parents``.

    Raises:
        Exception: If the repository cannot be opened or read.
    """
    try:
        # Open the repository
        repo = Repo(repo_path)

        # Extract commit data
        commits_data = []
        for commit in repo.iter_commits():
            commit_data = {
                "id": commit.hexsha,  # SHA1 commit ID
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "commit_date": commit.committed_datetime,
                "message": commit.message.strip(),
                "parents": [parent.hexsha for parent in commit.parents],
            }
            commits_data.append(commit_data)

        # Create DataFrame
        df = pd.DataFrame(commits_data)

        # Sort by commit date descending
        df = df.sort_values("commit_date", ascending=False)

        # Reset index
        df = df.reset_index(drop=True)

        log.info(f"Successfully extracted {len(df)} commits from repository")
        return df

    except Exception as e:
        log.error(f"Error extracting Git commits: {str(e)}")
        raise


def save_dataframe_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save a pandas DataFrame to a Parquet file.

    Creates parent directories as needed (unless the path is an S3 URI).

    Args:
        df: The DataFrame to persist.
        output_path: Local filesystem path or S3 URI for the Parquet file.

    Raises:
        Exception: If the file cannot be written.
    """
    try:
        if 's3://' not in output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save to Parquet
        df.to_parquet(output_path, index=False)
        log.info(f"DataFrame saved to {output_path}")

    except Exception as e:
        log.error(f"Error saving Parquet file: {str(e)}")
        raise


def main() -> None:
    """CLI entry point — parse args, extract commits, and save to Parquet."""
    parser = argparse.ArgumentParser(description="Convert Git commit history to pandas DataFrame", prog="git-to-pandas")

    # Add short arguments
    parser.add_argument("-r", "--repo-path", help="Path to the Git repository", required=True)
    parser.add_argument("-o", "--output-file", help="Output Parquet file path", required=True)

    args = parser.parse_args()

    try:
        # Validate repository path
        if not os.path.exists(args.repo_path):
            raise ValueError(f"Repository path does not exist: {args.repo_path}")

        # Extract commits
        df = extract_git_commits(args.repo_path)

        # Save to Parquet
        save_dataframe_to_parquet(df, args.output_file)

        log.info("Git commit to DataFrame conversion completed successfully")

    except Exception as e:
        log.error(f"Failed to convert Git commits to DataFrame: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
