#!/usr/bin/env python3
"""
Git Repository to Markdown Dataset Builder

This module analyzes a Git repository and generates markdown representations
of each commit's diff, storing them in a pandas DataFrame under the column 'git_diff_md'.
"""

import sys
import argparse
import pandas as pd
from git import Repo
from oats.log import cl

log = cl('build_git_dataset')


def build_git_diff_markdown(diff_index):
    """
    Convert a Git diff index to markdown format.

    Args:
        diff_index: Git diff index object

    Returns:
        str: Markdown formatted diff content
    """
    if not diff_index:
        return ""

    markdown_lines = []
    markdown_lines.append("| File | Type | Changes |")
    markdown_lines.append("|------|------|---------|")

    for diff in diff_index:
        # Get the file names
        old_file = diff.a_path if diff.a_path else "N/A"
        new_file = diff.b_path if diff.b_path else "N/A"

        # Determine type of change
        if diff.deleted_file:
            change_type = "Deleted"
        elif diff.renamed_file:
            change_type = "Renamed"
        elif diff.new_file:
            change_type = "Added"
        else:
            change_type = "Modified"

        # Get changes count
        additions = diff.added_lines if hasattr(diff, "added_lines") else 0
        deletions = diff.removed_lines if hasattr(diff, "removed_lines") else 0
        changes = f"+{additions} -{deletions}" if additions or deletions else "No changes"

        # Format row
        if old_file != new_file and diff.renamed_file:
            file_info = f"{old_file} → {new_file}"
        else:
            file_info = new_file if new_file else old_file

        markdown_lines.append(f"| {file_info} | {change_type} | {changes} |")

    return "\n".join(markdown_lines)


def extract_commit_info(commit):
    """
    Extract commit information.

    Args:
        commit: Git commit object

    Returns:
        dict: Commit information
    """
    return {
        "commit_hash": commit.hexsha,
        "author": commit.author.name,
        "email": commit.author.email,
        "date": commit.committed_datetime.isoformat(),
        "message": commit.message.strip(),
    }


def build_git_dataset(repo_path: str) -> pd.DataFrame:
    """
    Build a pandas DataFrame containing git commit diffs in markdown format.

    Args:
        repo_path: Path to the Git repository

    Returns:
        pd.DataFrame: DataFrame with commit information and markdown diffs
    """
    try:
        repo = Repo(repo_path)
        commits = list(repo.iter_commits())

        # Prepare data structures
        commit_data = []

        # Process commits from newest to oldest
        for i, commit in enumerate(commits):
            log.info(f"Processing commit {i + 1}/{len(commits)}: {commit.hexsha[:8]}")

            # Get commit info
            commit_info = extract_commit_info(commit)

            # Get diff for this commit (if not the first commit)
            if i < len(commits) - 1:
                # Get diff between this commit and the previous one
                prev_commit = commits[i + 1]
                diff_index = prev_commit.diff(commit, create_patch=True)
            else:
                # For the first commit, get diff with no parent
                diff_index = commit.diff(None, create_patch=True)

            # Convert diff to markdown
            markdown_diff = build_git_diff_markdown(diff_index)

            # Add to commit data
            commit_info["git_diff_md"] = markdown_diff
            commit_data.append(commit_info)

        # Create DataFrame
        df = pd.DataFrame(commit_data)
        return df

    except Exception as e:
        log.error(f"Error processing Git repository: {str(e)}")
        raise


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Build a markdown dataset from Git repository commits")
    parser.add_argument('-r', "--repo-path", default="/opt/ds/oats", help="Path to the Git repository (default: /opt/ds/oats)")
    parser.add_argument('-o', "--output-file", help="Output file path for the pandas DataFrame (optional)")

    args = parser.parse_args()

    try:
        log.info(f"Building Git dataset from repository: {args.repo_path}")
        df = build_git_dataset(args.repo_path)

        if args.output_file:
            df.to_parquet(args.output_file)
            log.info(f"DataFrame saved to: {args.output_file}")

        # Display summary
        log.info(f"Processed {len(df)} commits")
        log.info("Sample of git_diff_md column:")
        for idx, row in df.head().iterrows():
            print(f"\nCommit {row['commit_hash'][:8]} - {row['message'][:50]}...")
            print(row["git_diff_md"][:200] + "..." if len(row["git_diff_md"]) > 200 else row["git_diff_md"])

        return df

    except Exception as e:
        log.error(f"Failed to build Git dataset: {str(e)}")
        sys.exit(1)


# High-level API function compatible with the specified signature
def new_api(areq):
    """
    High-level API function for integrating with AgentReq.

    Args:
        areq: AgentReq object containing configuration

    Returns:
        pd.DataFrame: DataFrame with git_diff_md column
    """
    # This would integrate with the AgentReq object as needed
    # For now, we'll use the default behavior
    repo_path = getattr(areq, "repo_path", ".")
    return build_git_dataset(repo_path)


if __name__ == "__main__":
    # Run main function if executed directly
    df = main()
