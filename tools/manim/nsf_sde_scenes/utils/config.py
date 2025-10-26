"""
Configuration settings for video generation.
"""

import os
from pathlib import Path

class Config:
    """Configuration class for video generation settings."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    SITE_DIR = PROJECT_ROOT / "site"
    ASSETS_DIR = SITE_DIR / "assets"
    VIDEOS_DIR = ASSETS_DIR / "videos"
    IMAGES_DIR = ASSETS_DIR / "images"
    
    # Video settings
    VIDEO_WIDTH = 1280
    VIDEO_HEIGHT = 720
    VIDEO_FPS = 30
    VIDEO_DURATION = 10  # seconds
    VIDEO_QUALITY = 'high'
    
    # Colors (matching the website theme)
    PRIMARY_COLOR = '#2563eb'
    SECONDARY_COLOR = '#1e40af'
    BACKGROUND_COLOR = '#f8fafc'
    TEXT_COLOR = '#1f2937'
    
    # Font settings
    FONT_SIZE_TITLE = 48
    FONT_SIZE_SUBTITLE = 32
    FONT_SIZE_BODY = 24
    
    # Animation settings
    FADE_DURATION = 1.0  # seconds
    TRANSITION_DURATION = 0.5  # seconds
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist."""
        cls.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
