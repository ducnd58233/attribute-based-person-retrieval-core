"""SOLIDER Swin-B attribute classifier (pretrain-only LUPerson checkpoint).

Compliance (UPAR Track 2 / E21):
Use the **pretrain-only** LUPerson self-supervised Swin-B release from the
upstream SOLIDER repo — never a PAR-finetuned checkpoint from
``SOLIDER-PersonAttributeRecognition`` (or any other downstream finetune).

Provenance (do not commit weights into git):

- **Source README / Models table:**
  https://github.com/tinyvision/SOLIDER
- **Google Drive (Swin Base column of the Models table):**
  https://drive.google.com/file/d/1uh7tO34tMf73MJfFqyFEGx42UBktTbZU/view?usp=share_link
- **gdown / uc id:** ``1uh7tO34tMf73MJfFqyFEGx42UBktTbZU``
- **Expected local filename:** ``solider_swin_base.pth``
  (matches ``SOLIDER-PersonAttributeRecognition``'s
  ``./pretrained/solider_swin_base.pth`` naming; that repo loads the *same*
  pretrain file then finetunes — we stop at the pretrain file).
- **Confirmation it is pretrain-only:** upstream README states
  "All the models are trained on the whole LUPerson dataset" for the Models
  table links; PAR mA numbers in that table are *downstream* finetune results
  using those same pretrain weights, not the Drive blobs themselves.
- **Semantic weight (PAR docs):** ``0.8``
  (https://github.com/tinyvision/SOLIDER-PersonAttributeRecognition —
  "The semantic weight is set to 0.8 in these experiments.")
- **sha256:** not published by upstream. Optional ``model.weights_sha256`` in
  config verifies a local/cached file when you set it after first download.

Config knobs:

- ``model.weights_path`` — local ``.pth`` (skip download).
- ``model.pretrained_url`` — optional override URL (gdown id URL or http(s)).
- ``model.weights_sha256`` — optional hex digest for provenance check.
- ``model.semantic_weight`` — default ``0.8``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from torch import nn

from src.wacv.config import ModelConfig
from src.wacv.models._common import classification_head
from src.wacv.models.solider_backbone import swin_base_patch4_window7_224

LOGGER = logging.getLogger(__name__)

# --- Provenance constants (pretrain-only LUPerson SOLIDER Swin-B) ---
SOLIDER_SWIN_B_DRIVE_FILE_ID = "1uh7tO34tMf73MJfFqyFEGx42UBktTbZU"
SOLIDER_SWIN_B_DRIVE_VIEW_URL = f"https://drive.google.com/file/d/{SOLIDER_SWIN_B_DRIVE_FILE_ID}/view?usp=share_link"
SOLIDER_SWIN_B_GDOWN_URL = (
    f"https://drive.google.com/uc?id={SOLIDER_SWIN_B_DRIVE_FILE_ID}"
)
SOLIDER_SWIN_B_EXPECTED_FILENAME = "solider_swin_base.pth"
SOLIDER_SEMANTIC_WEIGHT_PAR = 0.8
SOLIDER_FEATURE_DIM = 1024  # Swin-B embed_dims=128 → stage-4 channels 1024
_DEFAULT_CACHE_DIRNAME = "solider"


class SoliderAttributeClassifier(nn.Module):
    """SOLIDER Swin backbone + linear attribute head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        feature_dim: int,
        image_size: tuple[int, int],
        semantic_weight: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.semantic_weight = semantic_weight
        # Lets HardQueryAdapter infer dim the same way as ConvNeXt / CLIP.
        self.classifier = head

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled Swin features (semantic controller applied inside backbone)."""
        features, _outs = self.backbone(images)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Attribute logits."""
        return self.head(self.encode_features(images))

    def freeze_backbone(self) -> None:
        """Freeze the vision tower (linear-probe stage)."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the vision tower (full fine-tune stage)."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights_sha256(path: Path, expected: str | None) -> str:
    """Return file sha256; raise if ``expected`` is set and mismatches."""
    actual = _sha256_file(path)
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(
            f"SOLIDER checkpoint sha256 mismatch for {path}: "
            f"got {actual}, expected {expected}. Refusing to load — "
            "confirm this is the pretrain-only LUPerson Swin-B release "
            f"({SOLIDER_SWIN_B_DRIVE_VIEW_URL}), not a PAR-finetuned file."
        )
    return actual


def _looks_like_gdrive(url: str) -> bool:
    if "drive.google.com" in url:
        return True
    return re.fullmatch(r"[A-Za-z0-9_-]{20,}", url) is not None


def _gdrive_file_id(url_or_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
        return url_or_id
    match = re.search(r"/file/d/([^/]+)/", url_or_id)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([^&]+)", url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"cannot parse Google Drive file id from {url_or_id!r}")


def download_solider_swin_b_checkpoint(
    destination: Path,
    *,
    url: str = SOLIDER_SWIN_B_GDOWN_URL,
) -> Path:
    """Download the official Swin-B pretrain checkpoint to ``destination``.

    Uses ``gdown`` for Google Drive links (dev dependency); plain
    ``urllib.request.urlretrieve`` for other http(s) URLs.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _looks_like_gdrive(url):
        file_id = _gdrive_file_id(url)
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "gdown is required to download the SOLIDER Google Drive "
                "checkpoint (install the project dev group: `uv sync --group "
                f"dev`). Manual fallback: download {SOLIDER_SWIN_B_DRIVE_VIEW_URL} "
                f"as {SOLIDER_SWIN_B_EXPECTED_FILENAME} and set model.weights_path."
            ) from exc
        gdown.download(id=file_id, output=str(destination), quiet=False)
    else:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported pretrained_url scheme: {url!r}")
        # URL is either the module constant or an explicit model.pretrained_url.
        urlretrieve(url, destination)
    if not destination.is_file():
        raise FileNotFoundError(f"download finished but missing file: {destination}")
    return destination


def resolve_solider_swin_b_weights(
    *,
    weights_path: Path | None,
    pretrained: bool,
    pretrained_url: str | None,
    weights_sha256: str | None,
    cache_dir: Path | None = None,
) -> Path | None:
    """Resolve a local SOLIDER Swin-B ``.pth``, downloading when needed.

    Returns ``None`` when ``pretrained`` is False and no ``weights_path`` is
    set (random backbone init). Raises if ``pretrained`` is True but the
    checkpoint cannot be found or downloaded.
    """
    if weights_path is not None:
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"model.weights_path does not exist: {weights_path}. "
                f"Expected the pretrain-only file named "
                f"{SOLIDER_SWIN_B_EXPECTED_FILENAME} from "
                f"{SOLIDER_SWIN_B_DRIVE_VIEW_URL}."
            )
        verify_weights_sha256(weights_path, weights_sha256)
        return weights_path

    if not pretrained:
        return None

    cache_root = cache_dir or (Path.home() / ".cache" / "wacv" / _DEFAULT_CACHE_DIRNAME)
    cached = cache_root / SOLIDER_SWIN_B_EXPECTED_FILENAME
    if cached.is_file():
        verify_weights_sha256(cached, weights_sha256)
        LOGGER.info("using cached SOLIDER Swin-B checkpoint %s", cached)
        return cached

    url = pretrained_url or SOLIDER_SWIN_B_GDOWN_URL
    LOGGER.info(
        "downloading SOLIDER Swin-B pretrain-only checkpoint from %s → %s",
        url,
        cached,
    )
    download_solider_swin_b_checkpoint(cached, url=url)
    actual = verify_weights_sha256(cached, weights_sha256)
    LOGGER.info(
        "SOLIDER Swin-B checkpoint ready path=%s sha256=%s provenance=%s",
        cached,
        actual,
        SOLIDER_SWIN_B_DRIVE_VIEW_URL,
    )
    return cached


def load_solider_swin_b_backbone(
    *,
    semantic_weight: float = SOLIDER_SEMANTIC_WEIGHT_PAR,
    weights_path: Path | None = None,
    img_size: tuple[int, int] = (256, 128),
) -> nn.Module:
    """Build Swin-B with semantic controller; optionally load pretrain weights."""
    backbone = swin_base_patch4_window7_224(
        img_size=img_size,
        convert_weights=False,
        semantic_weight=semantic_weight,
    )
    if weights_path is not None:
        backbone.init_weights(str(weights_path))
    return backbone


def build_solider_swin_b(
    config: ModelConfig,
    *,
    num_attributes: int,
    image_height: int = 256,
    image_width: int = 128,
    cache_dir: Path | None = None,
) -> SoliderAttributeClassifier:
    """Build SOLIDER Swin-B + attribute head (ConvNeXt-style recipe geometry)."""
    semantic_weight = float(config.semantic_weight)
    resolved = resolve_solider_swin_b_weights(
        weights_path=config.weights_path,
        pretrained=bool(config.pretrained),
        pretrained_url=config.pretrained_url,
        weights_sha256=config.weights_sha256,
        cache_dir=cache_dir,
    )
    backbone = load_solider_swin_b_backbone(
        semantic_weight=semantic_weight,
        weights_path=resolved,
        img_size=(image_height, image_width),
    )
    feature_dim = int(backbone.num_features[-1])
    head = classification_head(feature_dim, num_attributes, float(config.dropout))
    return SoliderAttributeClassifier(
        backbone,
        head,
        feature_dim=feature_dim,
        image_size=(image_height, image_width),
        semantic_weight=semantic_weight,
    )
