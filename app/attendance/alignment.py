"""THE shared alignment function -- imported by both registration and live
recognition. Do not reimplement this anywhere else.

If enrollment and live inference ever align faces differently, embeddings
from the two paths land in different regions of AdaFace's embedding space
and every cosine similarity score becomes meaningless -- faces of the same
person won't match, and the system fails silently (no errors, just wrong/no
matches). See README's accuracy-assumptions section.

The 112x112 reference landmarks below are the canonical 5-point layout used
by ArcFace/InsightFace *and* AdaFace -- verified against AdaFace's own
training-time alignment code (third_party/AdaFace/face_alignment/mtcnn_pytorch/
src/align_trans.py: REFERENCE_FACIAL_POINTS, expanded to a 112x112 square via
get_reference_facial_points(default_square=True)). SCRFD's 5 landmarks are in
the same left-eye/right-eye/nose/left-mouth/right-mouth order, so no
reordering is needed between detector output and this reference.
"""
from __future__ import annotations

import numpy as np
import cv2

# fmt: off
REFERENCE_5PTS_112 = np.array([
    [38.29459953, 51.69630051],  # left eye
    [73.53179932, 51.50139999],  # right eye
    [56.02519989, 71.73660278],  # nose tip
    [41.54930115, 92.3655014],   # left mouth corner
    [70.72990036, 92.20410156],  # right mouth corner
], dtype=np.float32)
# fmt: on

ALIGNED_SIZE = 112


def _umeyama_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form least-squares similarity transform (rotation + uniform
    scale + translation, no shear) mapping src points onto dst points.

    Standard Umeyama (1991) solution -- the same algorithm
    skimage.transform.SimilarityTransform.estimate() uses internally, which
    is what InsightFace's own alignment (face_align.norm_crop) relies on.
    Implemented directly here to avoid a scikit-image dependency for one
    function.

    Returns a 2x3 affine matrix usable directly with cv2.warpAffine.
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)

    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1

    R = U @ np.diag(d) @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = (S * d).sum() / var_src if var_src > 1e-8 else 1.0
    t = dst_mean - scale * (R @ src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M


def align_face(image_bgr: np.ndarray, landmarks_5: np.ndarray) -> np.ndarray:
    """Warp a face to a canonical 112x112 BGR crop using its 5 landmarks
    (left eye, right eye, nose, left mouth, right mouth -- SCRFD's kps
    order). Same function for registration images and live-recognition
    crops -- never diverge these paths."""
    M = _umeyama_similarity_transform(landmarks_5, REFERENCE_5PTS_112)
    return cv2.warpAffine(image_bgr, M, (ALIGNED_SIZE, ALIGNED_SIZE), borderValue=0.0)


def normalize_for_model(aligned_bgr: np.ndarray) -> np.ndarray:
    """(bgr/255 - 0.5) / 0.5, HWC uint8 -> CHW float32. Matches AdaFace's
    own to_input() preprocessing (inference.py in the AdaFace repo) exactly
    -- AdaFace was trained on BGR input, not RGB."""
    img = aligned_bgr.astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    return img.transpose(2, 0, 1)  # HWC -> CHW
