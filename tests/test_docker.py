"""Tests that the Docker files keep the data and the key out of the image, and the
service on this machine alone. The image itself is built in CI."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_the_build_context_leaves_out_the_data_and_the_key() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "data" in ignored and ".env" in ignored


def test_the_image_copies_the_code_alone_and_runs_as_an_ordinary_user() -> None:
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    copies = [line.split()[1] for line in lines if line.startswith("COPY ")]
    assert copies == ["pyproject.toml", "src"]
    users = [line.split()[1] for line in lines if line.startswith("USER ")]
    assert users == ["daleel"]


def test_compose_mounts_the_data_read_only_and_listens_on_this_machine_alone() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"gpu", "cpu"}
    for name, service in services.items():
        assert service["profiles"] == [name]
        assert all(port.startswith("127.0.0.1:") for port in service["ports"]), name
        mounts = [volume.split(":") for volume in service["volumes"]]
        data = {source: rest for source, *rest in mounts if source.startswith("./data/")}
        assert data["./data/serve"] == ["/app/data/serve"]
        assert all(rest[-1] == "ro" for source, rest in data.items() if source != "./data/serve")
        # The key is passed in when the service starts, never written into the file.
        assert service["environment"]["GEMINI_API_KEY"].startswith("${GEMINI_API_KEY:?")
    gpu = services["gpu"]["deploy"]["resources"]["reservations"]["devices"]
    assert gpu == [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
