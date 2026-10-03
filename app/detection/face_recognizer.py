import os
import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Real-time face recognition class using InsightFace (SCRFD + ArcFace) buffalo_l.

    Matches face crops or full images against embeddings of images registered in data/known_faces/.
    """

    def __init__(
        self,
        known_faces_dir: str | Path = "data/known_faces",
        distance_threshold: float = 1.0,
        onnx_provider: str = "CPUExecutionProvider",
    ) -> None:
        self.known_faces_dir = Path(known_faces_dir)
        self.distance_threshold = distance_threshold

        logger.info("Initializing FaceRecognizer with InsightFace buffalo_l (SCRFD + ArcFace)...")
        from insightface.app import FaceAnalysis

        providers = [onnx_provider] if onnx_provider else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.4)

        self.known_embeddings: dict[str, np.ndarray] = {}
        self.last_mtime = 0.0
        self.load_known_faces()

    def load_known_faces(self) -> None:
        """Scan known_faces directory and generate ArcFace embeddings for all registered faces."""
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created known faces directory at %s", self.known_faces_dir)
            return

        self.known_embeddings.clear()
        self.last_mtime = self.known_faces_dir.stat().st_mtime if self.known_faces_dir.exists() else 0.0

        # Supported image extensions
        extensions = {".jpg", ".jpeg", ".png", ".webp"}

        for file_path in self.known_faces_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Replace underscores in filenames with spaces for user-friendly display
                name = file_path.stem.replace("_", " ").strip()

                try:
                    img = cv2.imread(str(file_path))
                    if img is None:
                        logger.warning("Could not read image %s", file_path)
                        continue

                    embedding = self._compute_embedding(img)
                    if embedding is not None:
                        self.known_embeddings[name] = embedding
                        logger.info("Registered face: '%s' from %s", name, file_path.name)
                except Exception as e:
                    logger.exception("Failed to process known face image %s: %s", file_path, e)

        logger.info("Loaded %d known face(s) successfully.", len(self.known_embeddings))

    def _compute_embedding(self, img: np.ndarray) -> np.ndarray | None:
        """Internal helper to compute a normalized ArcFace embedding for an image or face crop."""
        try:
            if img is None or img.size == 0:
                return None

            faces = self.app.get(img)
            if faces:
                best_face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
                emb = getattr(best_face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(best_face, "embedding", None)
                if emb is not None:
                    norm = np.linalg.norm(emb)
                    return (emb / norm) if norm > 0 else emb

            # Fallback for tight crops: run recognition model directly
            rec_model = self.app.models.get("recognition")
            if rec_model is not None:
                resized = cv2.resize(img, (112, 112))
                emb = rec_model.get_feat(resized)
                if emb is not None:
                    if len(emb.shape) > 1:
                        emb = emb.flatten()
                    norm = np.linalg.norm(emb)
                    return (emb / norm) if norm > 0 else emb

            return None
        except Exception as e:
            logger.error("Error computing face embedding: %s", e)
            return None

    def recognize(self, face_crop: np.ndarray) -> tuple[str, float]:
        """Compare face crop against known faces.

        Returns:
            tuple: (name, distance) where name is "Unknown" if distance exceeds threshold.
        """
        try:
            current_mtime = self.known_faces_dir.stat().st_mtime if self.known_faces_dir.exists() else 0.0
            if current_mtime > self.last_mtime:
                logger.info("Known faces directory modified. Reloading...")
                self.load_known_faces()
        except Exception as e:
            logger.error("Failed to check mtime of known_faces_dir: %s", e)

        if not self.known_embeddings:
            return "Unknown", 1.0

        embedding = self._compute_embedding(face_crop)
        if embedding is None:
            return "Unknown", 1.0

        best_name = "Unknown"
        best_dist = 2.0  # Euclidean distance range is [0, 2] for L2-normalized vectors

        for name, known_emb in self.known_embeddings.items():
            dist = float(np.linalg.norm(embedding - known_emb))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= self.distance_threshold:
            return best_name, best_dist

        return "Unknown", best_dist
