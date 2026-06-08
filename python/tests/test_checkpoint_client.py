"""Checkpoint client tests."""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.worker.checkpoint_client import CheckpointClient


def test_checkpoint_client_pulls_verified_local_checkpoint(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    registry.publish({"model_state_dict": {"weight": torch.tensor([3.0])}}, 5, 10)
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([4.0])}}, 6, 20)
    client = CheckpointClient(str(tmp_path))

    assert client.latest_version() == meta.version
    state = client.pull(meta.version)

    assert client.current_version == meta.version
    torch.testing.assert_close(state["model_state_dict"]["weight"], torch.tensor([4.0]))


def test_checkpoint_client_accepts_file_url(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([1.0])}}, 1, 1)

    client = CheckpointClient(tmp_path.as_uri())

    assert client.latest_version() == meta.version


def test_checkpoint_client_pulls_verified_http_checkpoint(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([2.0])}}, 2, 5)

    with _serve_directory(tmp_path) as endpoint:
        client = CheckpointClient(endpoint)

        assert client.latest_version() == meta.version
        state = client.pull(meta.version)

    assert client.current_version == meta.version
    torch.testing.assert_close(state["model_state_dict"]["weight"], torch.tensor([2.0]))


def test_checkpoint_client_rejects_hash_mismatch(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([1.0])}}, 1, 1)
    with open(registry.resolve_path(meta), "ab") as fh:
        fh.write(b"corruption")
    client = CheckpointClient(str(tmp_path))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        client.pull(meta.version)


def test_checkpoint_client_rejects_unknown_version(tmp_path: Path) -> None:
    client = CheckpointClient(str(tmp_path))

    with pytest.raises(KeyError, match="unknown checkpoint version"):
        client.pull(99)


def test_checkpoint_client_rejects_paths_outside_registry_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pt"
    torch.save({"model_state_dict": {}}, outside)
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    (registry_root / "index.jsonl").write_text(
        json.dumps(
            {
                "created_step": 1,
                "path": str(outside),
                "policy_version": 1,
                "sha256": "0" * 64,
                "version": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CheckpointClient(str(registry_root), verify_hash=False)

    with pytest.raises(ValueError, match="escapes registry root"):
        client.pull(1)


def test_checkpoint_client_rejects_empty_index_path(tmp_path: Path) -> None:
    (tmp_path / "index.jsonl").write_text(
        json.dumps(
            {
                "created_step": 1,
                "path": ".",
                "policy_version": 1,
                "sha256": "0" * 64,
                "version": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CheckpointClient(str(tmp_path), verify_hash=False)

    with pytest.raises(ValueError, match="invalid checkpoint index line"):
        client.latest_version()


def test_checkpoint_client_rejects_duplicate_index_versions(tmp_path: Path) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish({"model_state_dict": {"weight": torch.tensor([1.0])}}, 1, 1)
    with (tmp_path / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**meta.__dict__, "path": meta.path}) + "\n")
    client = CheckpointClient(str(tmp_path))

    with pytest.raises(ValueError, match="duplicate checkpoint version"):
        client.latest_version()


def test_checkpoint_client_rejects_invalid_index_metadata(tmp_path: Path) -> None:
    (tmp_path / "index.jsonl").write_text(
        json.dumps(
            {
                "created_step": -1,
                "path": "checkpoint.pt",
                "policy_version": -1,
                "sha256": "bad",
                "version": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = CheckpointClient(str(tmp_path), verify_hash=False)

    with pytest.raises(ValueError, match="invalid checkpoint index line"):
        client.latest_version()


def test_checkpoint_client_rejects_unsupported_endpoint() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint registry endpoint"):
        CheckpointClient("s3://example.invalid/checkpoints")


@contextmanager
def _serve_directory(root: Path) -> Iterator[str]:
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, message_format: str, *args: object) -> None:
        return
