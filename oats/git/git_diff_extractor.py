#!/usr/bin/env python3

"""
Git Commit Diff Extractor

A tool to extract git diffs from commits in the /opt/ds/oats repository,
store commit metadata and diffs in a pandas DataFrame, and save results to disk.
"""

import os
import sys
import argparse
import pandas as pd
from git import Repo
from datetime import datetime
from typing import List, Dict, Any
from oats.log import cl
from oats.api_req1 import AgentReq

log = cl('git_diff_extractor')


def get_git_repo(repo_path: str) -> Repo:
    """Initialize and return a git repository object."""
    try:
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        if not os.path.isdir(os.path.join(repo_path, '.git')):
            raise ValueError(f"Path is not a git repository: {repo_path}")
        return Repo(repo_path)
    except Exception as e:
        log.error(f"Failed to initialize git repository at {repo_path}: {str(e)}")
        raise


def extract_commit_data(repo: Repo, max_commits: int = None) -> List[Dict[str, Any]]:
    """Extract commit metadata and diffs from the repository."""

    commits_data = []

    try:
        # Get all commits in reverse chronological order
        commits = list(repo.iter_commits(reverse=True))

        # Limit number of commits if specified
        if max_commits:
            commits = commits[:max_commits]

        for i, commit in enumerate(commits):
            # Extract commit metadata
            commit_data = {'commit_hash': commit.hexsha, 'author_name': commit.author.name, 'author_email': commit.author.email, 'author_date': commit.authored_datetime.isoformat(), 'committer_name': commit.committer.name, 'committer_email': commit.committer.email, 'committer_date': commit.committed_datetime.isoformat(), 'message_summary': commit.summary, 'message_full': commit.message, 'diff_content': '', 'diff_size': 0, 'files_changed': len(commit.stats.files), 'insertions': commit.stats.total['insertions'], 'deletions': commit.stats.total['deletions']}

            # Generate diff content for this commit
            if i == 0:
                # For the first commit, we need to compare against the initial state
                diff_content = ""
                try:
                    # Get the diff for the first commit (compare to initial state)
                    diff = commit.diff(None, create_patch=True)
                    # Convert diff to string format properly
                    if hasattr(diff, '__iter__'):
                        # If diff is iterable, join the patches
                        diff_patches = []
                        for d in diff:
                            if hasattr(d, 'patch'):
                                diff_patches.append(d.patch)
                            else:
                                diff_patches.append(str(d))
                        diff_content = '\n'.join(diff_patches) if diff_patches else ""
                    else:
                        # Single diff object
                        if hasattr(diff, 'patch'):
                            diff_content = diff.patch
                        else:
                            diff_content = str(diff)
                except Exception as e:
                    log.warning(f"Could not generate diff for first commit {commit.hexsha}: {str(e)}")
                    diff_content = "Diff generation failed"
            else:
                # For subsequent commits, get diff against previous commit
                prev_commit = commits[i - 1]
                try:
                    diff = commit.diff(prev_commit, create_patch=True)
                    # Convert diff to string format properly
                    if hasattr(diff, '__iter__'):
                        # If diff is iterable, join the patches
                        diff_patches = []
                        for d in diff:
                            if hasattr(d, 'patch'):
                                diff_patches.append(d.patch)
                            else:
                                diff_patches.append(str(d))
                        diff_content = '\n'.join(diff_patches) if diff_patches else ""
                    else:
                        # Single diff object
                        if hasattr(diff, 'patch'):
                            diff_content = diff.patch
                        else:
                            diff_content = str(diff)
                except Exception as e:
                    log.warning(f"Could not generate diff for commit {commit.hexsha}: {str(e)}")
                    diff_content = "Diff generation failed"

            commit_data['diff_content'] = diff_content
            commit_data['diff_size'] = len(diff_content) if diff_content else 0

            commits_data.append(commit_data)

    except Exception as e:
        log.error(f"Error extracting commit data: {str(e)}")
        raise

    return commits_data


def create_dataframe(commits_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from commit data."""
    df = pd.DataFrame(commits_data)
    return df


def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        df.to_csv(output_file, index=False)
        log.info(f"Successfully saved DataFrame to {output_file}")
    except Exception as e:
        log.error(f"Failed to save DataFrame to {output_file}: {str(e)}")
        raise


def new_api(repo_path: str = None) -> Dict[str, Any]:
    """
    Main API function to extract git diffs from repository.

    Args:
        areq: AgentReq instance containing configuration

    Returns:
        Dictionary with extraction results
    """
    if repo_path is None:
        repo_path = '/opt/ds/oats'
    try:
        # Get repo path from AgentReq (default to /opt/ds/oats)
        repo_path = getattr(areq, 'repo_path', '/opt/ds/oats')
        log.info(f"Starting git diff extraction from {repo_path}")

        # Validate repo path
        if not os.path.exists(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Initialize git repo
        repo = get_git_repo(repo_path)

        # Extract commit data (limit to 100 commits if not specified)
        max_commits = getattr(areq, 'max_commits', 100)
        commits_data = extract_commit_data(repo, max_commits)

        # Create DataFrame
        df = create_dataframe(commits_data)

        # Save to CSV
        if getattr(areq, 'output_file', None) is not None:
            output_file = getattr(areq, 'output_file')
        else:
            output_file = f'git_diffs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        save_dataframe(df, output_file)

        # Return result
        result = {'success': True, 'message': f'Successfully extracted {len(df)} commits', 'output_file': output_file, 'commit_count': len(df), 'df': df}

        log.info(result['message'])
        return result

    except Exception as e:
        error_msg = f"Git diff extraction failed: {str(e)}"
        log.error(error_msg)
        return {'success': False, 'message': error_msg}


def setup_parser():
    """Set up argument parser for CLI usage."""
    parser = argparse.ArgumentParser(description='Git Commit Diff Extractor')
    parser.add_argument('-r', '--repo-path', default='/opt/ds/oats', help='Path to git repository (default: /opt/ds/oats)')
    parser.add_argument('-o', '--output-file', default=None, help='Output CSV file path')
    parser.add_argument('-m', '--max-commits', type=int, default=100, help='Maximum number of commits to extract (default: 100)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    return parser


def main():
    """Main entry point for CLI usage."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.verbose:
        log.info("Verbose mode enabled")

    # Create minimal AgentReq for testing
    agent_req = AgentReq(prompt="Git diff extraction", db_enabled=False, s3_enabled=False, rc_enabled=False, dbui_enabled=False)

    # Set properties from CLI args
    setattr(agent_req, 'repo_path', args.repo_path)
    setattr(agent_req, 'output_file', args.output_file)
    setattr(agent_req, 'max_commits', args.max_commits)

    # Execute the API
    result = new_api(repo_path='/opt/ds/oats')

    if result['success']:
        log.info(f"Extraction completed successfully. Output saved to {result['output_file']}")
        return 0
    else:
        log.error(f"Extraction failed: {result['message']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
