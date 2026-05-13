"""
Git worktree manager — creates isolated workspaces for sub-agents.

Each worktree is a separate checkout of the repository, allowing
sub-agents to modify files without affecting the main working tree.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from oats.core.id import generate_short_id
from oats.log import cl

log = cl("git.worktree")


class WorktreeManager:
    """
    Manages git worktrees for isolated sub-agent work.

    Worktrees are created under <repo>/.coder-worktrees/<id>/
    and can be cleaned up after the agent finishes.
    """

    def __init__(self, repo_dir: Path) -> None:
        """Initialize the worktree manager for the given repository.

        Args:
            repo_dir: Path to the root of the Git repository.
        """
        self._repo_dir = repo_dir
        self._worktrees_dir = repo_dir / ".coder-worktrees"

    async def create(self, branch_name: str | None = None) -> Path:
        """
        Create an isolated worktree.

        Args:
            branch_name: Optional branch name. If None, creates a detached worktree
                         from the current HEAD.

        Returns:
            Path to the new worktree directory.
        """
        worktree_id = generate_short_id()
        worktree_path = self._worktrees_dir / worktree_id
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        if branch_name:
            # Create worktree on a new branch
            cmd = f"git -C {self._repo_dir} worktree add -b {branch_name} {worktree_path}"
        else:
            # Create detached worktree from HEAD
            branch = f"coder-agent-{worktree_id}"
            cmd = f"git -C {self._repo_dir} worktree add -b {branch} {worktree_path}"

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error = stderr.decode().strip()
            raise RuntimeError(f"Failed to create worktree: {error}")

        log.info(f"created worktree: {worktree_path}")
        return worktree_path

    async def cleanup(self, worktree_path: Path) -> None:
        """
        Remove a worktree.

        Only removes if the worktree has no uncommitted changes.
        """
        if not worktree_path.exists():
            return

        # Check for changes first
        if await self.has_changes(worktree_path):
            log.warn(f"worktree has changes, not cleaning up: {worktree_path}")
            return

        # Remove the worktree
        cmd = f"git -C {self._repo_dir} worktree remove {worktree_path} --force"
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        # Also try to delete the branch
        branch_name = worktree_path.name
        if branch_name.startswith("coder-agent-"):
            cmd = f"git -C {self._repo_dir} branch -D coder-agent-{branch_name}"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

        log.info(f"cleaned up worktree: {worktree_path}")

    async def has_changes(self, worktree_path: Path) -> bool:
        """Check if a worktree has uncommitted changes."""
        cmd = f"git -C {worktree_path} status --porcelain"
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return bool(stdout.decode().strip())

    async def list_worktrees(self) -> list[Path]:
        """List all worktrees managed by this instance."""
        if not self._worktrees_dir.exists():
            return []
        return [
            p for p in self._worktrees_dir.iterdir()
            if p.is_dir()
        ]

    async def merge_back(
        self,
        worktree_path: Path,
        target_branch: str | None = None,
    ) -> str:
        """
        Merge worktree changes back to the main branch.

        Returns the merge commit message or error.
        """
        # Get the branch name of the worktree
        cmd = f"git -C {worktree_path} rev-parse --abbrev-ref HEAD"
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        worktree_branch = stdout.decode().strip()

        if not worktree_branch:
            return "Error: Could not determine worktree branch"

        # Commit any uncommitted changes in the worktree
        if await self.has_changes(worktree_path):
            cmd = (
                f"cd {worktree_path} && "
                f"git add -A && "
                f"git commit -m 'Agent changes from {worktree_branch}'"
            )
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

        # Merge into target branch
        if target_branch is None:
            # Get current branch of main repo
            cmd = f"git -C {self._repo_dir} rev-parse --abbrev-ref HEAD"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            target_branch = stdout.decode().strip()

        cmd = f"git -C {self._repo_dir} merge {worktree_branch} --no-edit"
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            return f"Merge failed: {stderr.decode().strip()}"

        log.info(f"merged {worktree_branch} into {target_branch}")
        return f"Merged {worktree_branch} into {target_branch}"
