"""Authenticated checkpoint HTTP server tests."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from hkrl.learner.checkpoint_http import CheckpointHttpServer
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.worker.checkpoint_client import CheckpointClient


def test_checkpoint_http_serves_hash_verified_registry_with_bearer_auth(
    tmp_path: Path,
) -> None:
    registry = CheckpointRegistry(str(tmp_path))
    meta = registry.publish(
        {"model_state_dict": {"weight": torch.tensor([7.0])}},
        policy_version=3,
        step=4,
    )

    with _running_server(tmp_path, auth_token="secret") as endpoint:
        client = CheckpointClient(endpoint, auth_token="secret")
        assert client.latest_version() == meta.version
        state = client.pull(meta.version)

        request = Request(endpoint + "index.jsonl", method="HEAD")
        request.add_header("Authorization", "Bearer secret")
        with urlopen(request, timeout=2.0) as response:
            assert response.status == 200
            assert response.read() == b""
            assert response.headers["Cache-Control"] == "no-store"

    torch.testing.assert_close(
        state["model_state_dict"]["weight"],
        torch.tensor([7.0]),
    )


@pytest.mark.parametrize("token", [None, "wrong"])
def test_checkpoint_http_rejects_missing_or_wrong_token(
    tmp_path: Path,
    token: str | None,
) -> None:
    CheckpointRegistry(str(tmp_path)).publish(
        {"model_state_dict": {"weight": torch.tensor([1.0])}},
        policy_version=0,
        step=0,
    )

    with _running_server(tmp_path, auth_token="secret") as endpoint:
        client = CheckpointClient(endpoint, auth_token=token)
        with pytest.raises(HTTPError) as error:
            client.latest_version()

    assert error.value.code == 401


@pytest.mark.parametrize(
    "path",
    [
        "../outside.pt",
        "%2e%2e/outside.pt",
        "notes.txt",
        "checkpoint_latest.pt",
        "",
    ],
)
def test_checkpoint_http_exposes_only_registry_files(
    tmp_path: Path,
    path: str,
) -> None:
    (tmp_path / "notes.txt").write_text("secret", encoding="utf-8")
    with (
        _running_server(tmp_path, auth_token=None) as endpoint,
        pytest.raises(HTTPError) as error,
    ):
        urlopen(endpoint + path, timeout=2.0)

    assert error.value.code == 404


def test_checkpoint_http_rejects_writes(tmp_path: Path) -> None:
    with _running_server(tmp_path, auth_token=None) as endpoint:
        request = Request(endpoint + "index.jsonl", data=b"bad", method="POST")
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2.0)

    assert error.value.code == 405
    assert not (tmp_path / "index.jsonl").exists()


def test_checkpoint_http_creates_registry_root_and_starts_idempotently(
    tmp_path: Path,
) -> None:
    root = tmp_path / "new-registry"
    server = CheckpointHttpServer(root, "127.0.0.1:0", auth_token=None)
    try:
        server.start()
        endpoint = server.endpoint
        server.start()
        assert root.is_dir()
        assert server.endpoint == endpoint
    finally:
        server.close()
        server.close()


class _running_server:
    def __init__(self, root: Path, *, auth_token: str | None) -> None:
        self.server = CheckpointHttpServer(
            root,
            "127.0.0.1:0",
            auth_token=auth_token,
        )
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        self.thread = self.server.serve_in_thread()
        assert self.server.endpoint is not None
        return f"http://{self.server.endpoint}/"

    def __exit__(self, *exc_info: object) -> None:
        self.server.close()
        assert self.thread is not None
        self.thread.join(timeout=3.0)
        assert not self.thread.is_alive()
