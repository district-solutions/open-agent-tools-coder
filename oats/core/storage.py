"""
File-based JSON storage system with locking.
"""

from __future__ import annotations

import json
import asyncio
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Any, TypeVar, Generic
from dataclasses import dataclass, field
import fcntl

from pydantic import BaseModel

from oats.core.config import get_data_dir


T = TypeVar("T", bound=BaseModel)


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


@dataclass
class Storage(Generic[T]):
    """
    Generic file-based storage for Pydantic models.

    Provides CRUD operations with file locking for concurrent access.
    """

    namespace: str
    model_class: type[T]
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def storage_dir(self) -> Path:
        """Get the storage directory for this namespace."""
        return get_data_dir() / self.namespace

    def _get_file_path(self, id: str) -> Path:
        """Get the file path for an item."""
        return self.storage_dir / f"{id}.json"

    async def _ensure_dir(self) -> None:
        """Ensure the storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def create(self, id: str, data: T) -> T:
        """Create a new item."""
        async with self._lock:
            await self._ensure_dir()
            file_path = self._get_file_path(id)

            if file_path.exists():
                raise StorageError(f"Item with id '{id}' already exists")

            await self._write_file(file_path, data)
            return data

    async def read(self, id: str) -> T | None:
        """Read an item by ID."""
        file_path = self._get_file_path(id)

        if not file_path.exists():
            return None

        return await self._read_file(file_path)

    async def update(self, id: str, data: T) -> T:
        """Update an existing item."""
        async with self._lock:
            file_path = self._get_file_path(id)

            if not file_path.exists():
                raise StorageError(f"Item with id '{id}' not found")

            await self._write_file(file_path, data)
            return data

    async def upsert(self, id: str, data: T) -> T:
        """Create or update an item."""
        async with self._lock:
            await self._ensure_dir()
            file_path = self._get_file_path(id)
            await self._write_file(file_path, data)
            return data

    async def delete(self, id: str) -> bool:
        """Delete an item. Returns True if deleted, False if not found."""
        async with self._lock:
            file_path = self._get_file_path(id)

            if not file_path.exists():
                return False

            await aiofiles.os.remove(file_path)
            return True

    async def list(self) -> list[T]:
        """List all items."""
        await self._ensure_dir()

        items: list[T] = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                item = await self._read_file(file_path)
                if item:
                    items.append(item)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")

        return items

    async def list_ids(self) -> list[str]:
        """List all item IDs."""
        await self._ensure_dir()
        return [f.stem for f in self.storage_dir.glob("*.json")]

    async def _read_file(self, file_path: Path) -> T | None:
        """Read and parse a JSON file."""
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                return self.model_class.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            return None

    async def _write_file(self, file_path: Path, data: T) -> None:
        """Write a Pydantic model to a JSON file with locking."""
        json_str = data.model_dump_json(indent=2)

        async with aiofiles.open(file_path, "w") as f:
            # Get file descriptor for locking
            fd = f.fileno()
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                await f.write(json_str)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)


class KeyValueStorage:
    """
    Simple key-value storage for arbitrary JSON data.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = asyncio.Lock()

    @property
    def file_path(self) -> Path:
        """Get the storage file path."""
        return get_data_dir() / f"{self.name}.json"

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        data = await self._read()
        return data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value."""
        async with self._lock:
            data = await self._read()
            data[key] = value
            await self._write(data)

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if deleted."""
        async with self._lock:
            data = await self._read()
            if key in data:
                del data[key]
                await self._write(data)
                return True
            return False

    async def all(self) -> dict[str, Any]:
        """Get all data."""
        return await self._read()

    async def _read(self) -> dict[str, Any]:
        """Read the storage file."""
        if not self.file_path.exists():
            return {}

        try:
            async with aiofiles.open(self.file_path, "r") as f:
                content = await f.read()
                return json.loads(content) if content else {}
        except (json.JSONDecodeError, OSError):
            return {}

    async def _write(self, data: dict[str, Any]) -> None:
        """Write to the storage file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(self.file_path, "w") as f:
            await f.write(json.dumps(data, indent=2))
