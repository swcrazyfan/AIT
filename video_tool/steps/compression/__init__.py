# This file makes the 'compression' directory a Python package.

from .ffmpeg_compress import FFmpegCompressStep

__all__ = [
    "FFmpegCompressStep"
]