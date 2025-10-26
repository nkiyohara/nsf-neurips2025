"""
Video generation utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import imageio

from .config import Config

class VideoGenerator:
    """Utility class for video generation operations."""
    
    def __init__(self, width: int = None, height: int = None, fps: int = None):
        self.width = width or Config.VIDEO_WIDTH
        self.height = height or Config.VIDEO_HEIGHT
        self.fps = fps or Config.VIDEO_FPS
        self.config = Config()
        
    def create_blank_frame(self, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Create a blank frame with specified color."""
        frame = np.full((self.height, self.width, 3), color, dtype=np.uint8)
        return frame
    
    def add_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                 font_scale: float = 1.0, color: Tuple[int, int, int] = (0, 0, 0),
                 thickness: int = 2) -> np.ndarray:
        """Add text to a frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        return frame
    
    def create_fade_effect(self, frame1: np.ndarray, frame2: np.ndarray, 
                          alpha: float) -> np.ndarray:
        """Create fade effect between two frames."""
        return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
    
    def save_video(self, frames: List[np.ndarray], output_path: Path, 
                   codec: str = 'mp4v') -> None:
        """Save frames as video file."""
        Config.ensure_directories()
        
        # Use imageio for better compatibility
        with imageio.get_writer(str(output_path), fps=self.fps, codec=codec) as writer:
            for frame in frames:
                # Convert BGR to RGB for imageio
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                writer.append_data(frame_rgb)
    
    def resize_frame(self, frame: np.ndarray, target_width: int = None, 
                    target_height: int = None) -> np.ndarray:
        """Resize frame to target dimensions."""
        target_width = target_width or self.width
        target_height = target_height or self.height
        return cv2.resize(frame, (target_width, target_height))
    
    def create_gradient_background(self, start_color: Tuple[int, int, int], 
                                 end_color: Tuple[int, int, int], 
                                 direction: str = 'horizontal') -> np.ndarray:
        """Create gradient background."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if direction == 'horizontal':
            for i in range(self.width):
                alpha = i / self.width
                color = tuple(int(start_color[j] * (1 - alpha) + end_color[j] * alpha) 
                            for j in range(3))
                frame[:, i] = color
        else:  # vertical
            for i in range(self.height):
                alpha = i / self.height
                color = tuple(int(start_color[j] * (1 - alpha) + end_color[j] * alpha) 
                            for j in range(3))
                frame[i, :] = color
                
        return frame
