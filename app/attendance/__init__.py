"""New attendance pipeline: SCRFD detection, ByteTrack tracking, AdaFace
recognition, FAISS matching. Isolated from app/detection and app/tracking
(the legacy YOLO/facenet/ArcFace-via-external-API pipeline) -- AdaFace
embeddings are not compatible with those, so nothing here imports from there.
"""
