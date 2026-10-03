"""Convert the AdaFace IR-101 (WebFace12M) checkpoint to ONNX.

Gate script for the new attendance pipeline: if this fails, do not write any
application code that depends on the recognition model.

Usage:
    source .venv/bin/activate
    python scripts/export_adaface_onnx.py

Requires third_party/AdaFace (git clone of mk-minchul/AdaFace) and the
IR-101 WebFace12M checkpoint at
third_party/AdaFace/pretrained/adaface_ir101_webface12m.ckpt
(download link is in that repo's README, under Pretrained Models -> R100 / WebFace12M).
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
ADAFACE_REPO = ROOT / "third_party" / "AdaFace"
CHECKPOINT = ADAFACE_REPO / "pretrained" / "adaface_ir101_webface12m.ckpt"
ONNX_OUT = ROOT / "models" / "adaface_ir101_webface12m.onnx"

sys.path.insert(0, str(ADAFACE_REPO))
import net  # noqa: E402  (AdaFace repo module)

OPSET = 17
BATCH_CHECK = 32


class AdaFaceEmbedder(nn.Module):
    """Wraps the AdaFace backbone to expose a single 512-d embedding output.

    net.Backbone.forward returns (embedding, norm); norm is only used for
    AdaFace's training-time quality-adaptive margin and is dropped here since
    inference only needs the embedding.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _norm = self.backbone(x)
        return embedding


def load_backbone() -> nn.Module:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}. Download adaface_ir101_webface12m.ckpt "
            "(R100 / WebFace12M) from the AdaFace README and place it there."
        )
    backbone = net.build_model("ir_101")
    state = torch.load(CHECKPOINT, map_location="cpu")["state_dict"]
    # checkpoint keys are prefixed "model." (pytorch-lightning module wrapper)
    backbone_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()
    return backbone


def export(backbone: nn.Module) -> None:
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    model = AdaFaceEmbedder(backbone)
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_OUT),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,  # dynamic_axes is a legacy-exporter arg; dynamo=True needs onnxscript and uses dynamic_shapes instead
    )
    print(f"Exported ONNX model to {ONNX_OUT}")


def verify() -> None:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(ONNX_OUT), providers=providers)
    active = sess.get_providers()
    print("onnxruntime session providers:", active)
    if active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "CUDAExecutionProvider did not load for the exported model -- "
            "silently running on CPU. Did you `source .venv/bin/activate` "
            "(needed for the LD_LIBRARY_PATH CUDA-13 pip package fix)?"
        )

    x = np.random.randn(BATCH_CHECK, 3, 112, 112).astype(np.float32)
    (out,) = sess.run(None, {"input": x})
    print("output shape:", out.shape)
    assert out.shape == (BATCH_CHECK, 512), f"expected ({BATCH_CHECK}, 512), got {out.shape}"

    norms = np.linalg.norm(out, axis=1)
    print("embedding L2 norms (should be ~1.0, model normalizes internally):", norms[:5], "...")
    assert np.allclose(norms, 1.0, atol=1e-3), "embeddings are not unit-normalized as expected"

    print(f"GATE PASSED: ({BATCH_CHECK}, 512) output confirmed on CUDAExecutionProvider")


if __name__ == "__main__":
    print("Loading AdaFace IR-101 WebFace12M checkpoint...")
    backbone = load_backbone()
    print("Exporting to ONNX (opset", OPSET, ", dynamic batch axis)...")
    export(backbone)
    print("Verifying exported model...")
    verify()
