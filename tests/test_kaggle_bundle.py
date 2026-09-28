"""Tests for Kaggle single-file payload embedding."""

from __future__ import annotations

import ast
import base64
import io
import zipfile
from pathlib import Path

from src.wacv.remote.kaggle import embed_staging_payload
from src.wacv.remote.stage import build_staging_dir


def test_embed_staging_payload_extracts_scripts(tmp_path: Path) -> None:
    staging = tmp_path / "remote-kaggle"
    build_staging_dir(Path("configs/e19_clip_l14_lora.yml"), staging)
    assert (staging / "scripts" / "download-dataset.py").is_file()

    embed_staging_payload(staging)
    wrapper = (staging / "run_remote.py").read_text(encoding="utf-8")
    assert "PAYLOAD_B64" in wrapper
    assert "run_inner.py" in wrapper
    assert not (staging / "run_inner.py").exists()

    marker = "PAYLOAD_B64 = "
    start = wrapper.index(marker) + len(marker)
    end = wrapper.index("\n\n", start)
    payload_b64 = ast.literal_eval(wrapper[start:end])
    assert isinstance(payload_b64, str)

    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload_b64))) as archive:
        archive.extractall(extract)
    assert (extract / "scripts" / "download-dataset.py").is_file()
    assert (extract / "run_inner.py").is_file()
    assert (extract / "src" / "wacv" / "__init__.py").is_file()
