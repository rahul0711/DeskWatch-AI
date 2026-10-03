import av
import threading
import time
import numpy as np

class PyAVReader:
    """
    Decodes an RTSP stream using PyAV, maintaining a zero-buffer,
    latest-frame-only pipeline. Stale frames are continuously discarded.
    """
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="PyAVReaderThread")
        self.thread.start()
        return self

    def _run(self):
        print(f"Connecting to RTSP stream: {self.rtsp_url}")
        while self.running:
            try:
                # Open stream with low latency & no-buffer options
                container = av.open(
                    self.rtsp_url,
                    options={
                        'rtsp_transport': 'tcp',
                        'fflags': 'nobuffer',
                        'flags': 'low_delay',
                        'stimeout': '5000000'  # 5-second socket timeout
                    }
                )
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'  # Enable multi-threaded decoding
                
                # Decoded frame generation loop
                for frame in container.decode(stream):
                    if not self.running:
                        break
                    
                    # Convert decoded frame to RGB numpy array
                    rgb_image = frame.to_ndarray(format='rgb24')
                    
                    # Store only the latest frame, dropping older unread ones
                    with self.lock:
                        self.latest_frame = rgb_image
                        
                container.close()
            except Exception as e:
                print(f"PyAV Reader exception: {e}. Reconnecting in 2 seconds...")
                time.sleep(2)

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Returns the latest frame and clears it from memory
        to prevent duplicate frames from being processed.
        """
        with self.lock:
            frame = self.latest_frame
            self.latest_frame = None
            return frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        print("PyAV Reader stopped.")
