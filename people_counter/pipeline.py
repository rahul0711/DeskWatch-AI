import os
import time
import threading
import asyncio
import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from camera import PyAVReader

class RFDetrDetector:
    """
    ONNX Runtime wrapper for RF-DETR (Roboflow Detection Transformer).
    Supports swapping to TensorRT execution provider.
    """
    def __init__(self, weights_path: str, conf_threshold: float = 0.3, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        
        # Swappable providers based on config
        device_lower = device.lower().strip()
        if device_lower == "tensorrt":
            providers = ["TensorRTExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device_lower in ("cuda", "gpu"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            
        print(f"Loading RF-DETR ONNX model from {weights_path} with providers {providers}...")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights file not found at {weights_path}. "
                "Please place your head-trained RF-DETR ONNX model in this path."
            )
            
        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape # e.g. [1, 3, 640, 640]
        
        # Check model shape, default to 640 if dynamic
        self.input_h = input_shape[2] if (len(input_shape) > 2 and isinstance(input_shape[2], int)) else 640
        self.input_w = input_shape[3] if (len(input_shape) > 3 and isinstance(input_shape[3], int)) else 640
        print(f"RF-DETR ONNX model loaded. Expected input size: {self.input_w}x{self.input_h}")

    def detect(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs model inference.
        Returns:
            boxes: absolute coordinate bounding boxes (x1, y1, x2, y2)
            confidences: float confidence values
            class_ids: integer class ids
        """
        h_orig, w_orig = rgb_frame.shape[:2]
        
        # Resize to model expectation
        resized = cv2.resize(rgb_frame, (self.input_w, self.input_h))
        # HWC to CHW layout, normalize to [0, 1]
        input_data = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        # Add batch dimension [1, 3, H, W]
        input_tensor = np.expand_dims(input_data, axis=0)
        
        # Run inference session
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        boxes_out = None
        logits_out = None
        
        # Bind output tensors dynamically based on shapes
        for out in outputs:
            if len(out.shape) == 3 and out.shape[2] == 4:
                boxes_out = out[0]  # shape [N, 4]
            elif len(out.shape) == 3 and out.shape[2] > 4:
                logits_out = out[0] # shape [N, C]
                
        # Fallback if shape inspection fails
        if (boxes_out is None or logits_out is None) and len(outputs) >= 2:
            if outputs[0].shape[-1] == 4:
                boxes_out = outputs[0][0]
                logits_out = outputs[1][0]
            else:
                boxes_out = outputs[1][0]
                logits_out = outputs[0][0]
                
        if boxes_out is None or logits_out is None:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))
            
        # Apply sigmoid to logits to get probabilities
        scores_all = 1.0 / (1.0 + np.exp(-logits_out))
        class_ids_all = np.argmax(scores_all, axis=-1)
        confidences_all = np.max(scores_all, axis=-1)
        
        # Filter out low-confidence detections
        keep = confidences_all >= self.conf_threshold
        filtered_boxes = boxes_out[keep]
        filtered_conf = confidences_all[keep]
        filtered_classes = class_ids_all[keep]
        
        abs_boxes = []
        for box in filtered_boxes:
            # Check if boxes are already absolute or normalized
            val_max = np.max(box)
            if val_max <= 1.01:
                # Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
                cx, cy, w_box, h_box = box[0], box[1], box[2], box[3]
                x1 = (cx - w_box / 2.0) * w_orig
                y1 = (cy - h_box / 2.0) * h_orig
                x2 = (cx + w_box / 2.0) * w_orig
                y2 = (cy + h_box / 2.0) * h_orig
            else:
                # Assuming absolute [x1, y1, x2, y2] or [cx, cy, w, h]
                # If absolute cxcywh:
                # Let's assume standard absolute xyxy coordinates
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
            x1 = max(0.0, min(w_orig, x1))
            y1 = max(0.0, min(h_orig, y1))
            x2 = max(0.0, min(w_orig, x2))
            y2 = max(0.0, min(h_orig, y2))
            abs_boxes.append([x1, y1, x2, y2])
            
        return np.array(abs_boxes, dtype=np.float32), filtered_conf, filtered_classes


class PeopleCounterPipeline:
    """
    Orchestrates the counting pipeline: PyAV reader -> RF-DETR ONNX detection
    -> ByteTrack tracking -> LineZone cross checking -> WebSocket JSON dispatch.
    """
    def __init__(self, on_result_callback, loop: asyncio.AbstractEventLoop):
        self.on_result_callback = on_result_callback
        self.loop = loop
        
        # Load environment variables
        self.rtsp_url = os.getenv("RTSP_URL", "rtsp://localhost:8554/live")
        self.weights_path = os.getenv("WEIGHTS_PATH", "weights/rfdetr_head.onnx")
        self.conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.3"))
        self.device = os.getenv("DEVICE", "cpu")
        
        # Parse LineZone coordinates (format: x1,y1,x2,y2)
        line_coords_str = os.getenv("LINE_COORDS", "100,500,1100,500")
        try:
            coords = [int(x) for x in line_coords_str.split(",")]
            self.start_point = sv.Point(coords[0], coords[1])
            self.end_point = sv.Point(coords[2], coords[3])
        except Exception as e:
            print(f"Error parsing LINE_COORDS '{line_coords_str}': {e}. Falling back to default line.")
            self.start_point = sv.Point(100, 500)
            self.end_point = sv.Point(1100, 500)
            
        # Initialize pipeline components
        self.reader = PyAVReader(self.rtsp_url)
        self.detector = RFDetrDetector(
            weights_path=self.weights_path,
            conf_threshold=self.conf_threshold,
            device=self.device
        )
        self.tracker = sv.ByteTrack()
        self.line_zone = sv.LineZone(start=self.start_point, end=self.end_point)
        
        self.seen_track_ids = set()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.reader.start()
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="PipelineThread")
        self.thread.start()
        print("People Counter pipeline started in background thread.")

    def _run_loop(self):
        while self.running:
            frame = self.reader.get_latest_frame()
            if frame is None:
                time.sleep(0.005) # Prevent high CPU usage when no new frames are decoded
                continue
                
            try:
                # 1. Detect heads
                boxes, confidences, class_ids = self.detector.detect(frame)
                
                if len(boxes) == 0:
                    detections = sv.Detections.empty()
                else:
                    detections = sv.Detections(
                        xyxy=boxes,
                        confidence=confidences,
                        class_id=class_ids
                    )
                    
                # 2. Update ByteTrack
                detections = self.tracker.update_with_detections(detections)
                
                # 3. Update LineZone counters
                self.line_zone.trigger(detections=detections)
                
                # 4. Handle per-identity work (ONCE per track ID)
                if detections.tracker_id is not None and len(detections.tracker_id) > 0:
                    for tracker_id in detections.tracker_id:
                        tid = int(tracker_id)
                        if tid not in self.seen_track_ids:
                            self.seen_track_ids.add(tid)
                            self._on_new_track(tid)
                            
                    # 5. Broadcast track results to WebSocket handler
                    for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):
                        result_dict = {
                            "track_id": int(tracker_id),
                            "bbox": [float(val) for val in bbox],
                            "in_count": int(self.line_zone.in_count),
                            "out_count": int(self.line_zone.out_count)
                        }
                        self.on_result_callback(result_dict, self.loop)
            except Exception as e:
                print(f"Error in pipeline iteration: {e}")
                time.sleep(0.1)

    def _on_new_track(self, track_id: int):
        """Runs custom work ONCE per new track ID."""
        print(f"[Identity Work] New head detected and tracked with ID: {track_id}")

    def stop(self):
        self.running = False
        self.reader.stop()
        if self.thread:
            self.thread.join(timeout=3)
        print("People Counter pipeline stopped.")
