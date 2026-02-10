"""Video I/O using ffmpeg subprocess pipes.

Adapted from ReCoNet/ffmpeg_tools.py.
"""

import json
import shlex
from math import ceil
from subprocess import DEVNULL, PIPE, Popen, check_output

import numpy as np


def _check_wait(p):
    status = p.wait()
    if status != 0:
        raise RuntimeError(f"{p.args} returned non-zero status {status}")


def _ffprobe(filepath: str, cmd: str = "ffprobe") -> dict:
    command = f"{cmd} -loglevel fatal -print_format json -show_format -show_streams {filepath}"
    output = check_output(shlex.split(command))
    return json.loads(output)


def _fraction(s: str | None) -> float | None:
    if s is None:
        return None
    num, den = s.split("/")
    return int(num) / int(den)


class _VideoIterator:
    def __init__(self, reader: "VideoReader"):
        self._reader = reader
        self._closed = False

        cmd_parts = [
            f"{reader.ffmpeg_cmd} -loglevel error -y -nostdin",
            f"-i {reader.filepath}",
        ]
        if reader.fps is not None:
            cmd_parts.append(f"-filter fps=fps={reader.fps}:round=up")
        cmd_parts.append(f"-f rawvideo -pix_fmt rgb24 pipe:")
        cmd = " ".join(cmd_parts)

        self._ffmpeg = Popen(shlex.split(cmd), stdout=PIPE, stdin=DEVNULL)

    def __next__(self) -> np.ndarray:
        frame_size = self._reader.width * self._reader.height * 3
        in_bytes = self._ffmpeg.stdout.read(frame_size)

        if len(in_bytes) == 0:
            self._close()
            raise StopIteration()

        assert len(in_bytes) == frame_size
        return np.frombuffer(in_bytes, np.uint8).reshape(
            (self._reader.height, self._reader.width, 3)
        )

    def __iter__(self):
        return self

    def __del__(self):
        self._close()

    def _close(self):
        if not self._closed:
            self._closed = True
            self._ffmpeg.kill()


class VideoReader:
    """Read video frames as numpy arrays using ffmpeg."""

    def __init__(
        self,
        filepath: str,
        fps: float | None = None,
        ffmpeg_cmd: str = "ffmpeg",
        ffprobe_cmd: str = "ffprobe",
    ):
        probe = _ffprobe(filepath, cmd=ffprobe_cmd)
        stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

        self.width = int(stream["width"])
        self.height = int(stream["height"])
        self.fps = fps or _fraction(stream.get("r_frame_rate"))
        self.duration = float(stream.get("duration", 0))

        self.filepath = filepath
        self.ffmpeg_cmd = ffmpeg_cmd

    def __iter__(self):
        return _VideoIterator(self)


class VideoWriter:
    """Write video frames from numpy arrays using ffmpeg."""

    def __init__(
        self,
        filepath: str,
        input_width: int,
        input_height: int,
        input_fps: float,
        output_width: int | None = None,
        output_height: int | None = None,
        output_format: str = "yuv420p",
        ffmpeg_cmd: str = "ffmpeg",
    ):
        self.filepath = filepath
        self.input_width = input_width
        self.input_height = input_height
        self.input_fps = input_fps
        self.output_width = output_width
        self.output_height = output_height
        self.output_format = output_format
        self.ffmpeg_cmd = ffmpeg_cmd
        self._ffmpeg = None

    def __enter__(self):
        cmd_parts = [
            f"{self.ffmpeg_cmd} -y -loglevel error",
            f"-f rawvideo -pix_fmt rgb24 -video_size {self.input_width}x{self.input_height} -framerate {self.input_fps} -i pipe:",
            f"-pix_fmt {self.output_format}",
        ]
        if self.output_width is not None and self.output_height is not None:
            cmd_parts.append(f"-s {self.output_width}x{self.output_height}")
        cmd_parts.append(self.filepath)
        cmd = " ".join(cmd_parts)

        self._ffmpeg = Popen(shlex.split(cmd), stdin=PIPE)
        return self

    def write(self, frame: np.ndarray):
        assert (
            frame.dtype == np.uint8
            and frame.ndim == 3
            and frame.shape == (self.input_height, self.input_width, 3)
        )
        self._ffmpeg.stdin.write(frame.tobytes())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ffmpeg:
            self._ffmpeg.stdin.close()
            _check_wait(self._ffmpeg)
