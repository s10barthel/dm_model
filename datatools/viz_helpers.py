from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


SeriesOrFrame = pd.Series | pd.DataFrame


def figure_to_rgb_image(fig, dpi: int = 150, tight: bool = True) -> Image.Image:
    buffer = io.BytesIO()
    try:
        savefig_kwargs = {"format": "png", "dpi": dpi}
        if tight:
            savefig_kwargs["bbox_inches"] = "tight"
        fig.savefig(buffer, **savefig_kwargs)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return image.convert("RGB").copy()
    finally:
        buffer.close()


def _union_indexes(values: list[SeriesOrFrame]) -> tuple[pd.Index, pd.Index | None]:
    first = values[0]
    index = first.index
    columns = getattr(first, "columns", None)
    for value in values[1:]:
        index = index.union(value.index)
        if columns is not None:
            columns = columns.union(value.columns)
    return index, columns


def _align_like(values: list[SeriesOrFrame]) -> list[SeriesOrFrame]:
    if not values:
        raise ValueError("Expected at least one pandas object to align.")

    all_series = all(isinstance(value, pd.Series) for value in values)
    all_frames = all(isinstance(value, pd.DataFrame) for value in values)
    if not all_series and not all_frames:
        raise TypeError("All pass-score inputs must be either Series or DataFrames.")

    index, columns = _union_indexes(values)
    if all_series:
        return [value.reindex(index) for value in values]
    return [value.reindex(index=index, columns=columns) for value in values]


def compute_pass_score(
    pass_success: SeriesOrFrame,
    outcome_scoring_success: SeriesOrFrame,
    outcome_scoring_failure: SeriesOrFrame,
    outcome_conceding_success: SeriesOrFrame,
    outcome_conceding_failure: SeriesOrFrame,
) -> SeriesOrFrame:
    aligned = _align_like(
        [
            pass_success,
            outcome_scoring_success,
            outcome_scoring_failure,
            outcome_conceding_success,
            outcome_conceding_failure,
        ]
    )
    aligned_pass_success, scoring_success, scoring_failure, conceding_success, conceding_failure = aligned
    pass_score = aligned_pass_success * (scoring_success - conceding_success) + (1.0 - aligned_pass_success) * (
        scoring_failure - conceding_failure
    )
    if isinstance(pass_score, pd.Series):
        pass_score.name = getattr(pass_success, "name", None)
    return pass_score


def _save_gif(images: list[Image.Image], output_path: Path, fps: float) -> None:
    duration_ms = max(1, int(round(1000 / fps)))
    first_frame = images[0].convert("RGB").quantize(colors=255, dither=Image.Dither.NONE)
    palette_source = first_frame.copy()
    palettized_frames = [first_frame]
    for image in images[1:]:
        palettized_frames.append(image.convert("RGB").quantize(palette=palette_source, dither=Image.Dither.NONE))

    palettized_frames[0].save(
        output_path,
        save_all=True,
        append_images=palettized_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _save_mp4(images: Iterable[Image.Image], output_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for MP4 output but was not found on PATH.")

    iterator = iter(images)
    try:
        first_image = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot save an animation with no frames.") from exc

    first_frame = _prepare_mp4_frame(first_image)
    frame_size = first_frame.size
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frame_size[0]}x{frame_size[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        process.stdin.write(first_frame.tobytes())
        for image in iterator:
            frame = _prepare_mp4_frame(image)
            if frame.size != frame_size:
                raise ValueError(
                    f"All frames in an MP4 animation must share the same size. Expected {frame_size}, got {frame.size}."
                )
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {output_path.name}: {stderr or 'unknown error'}")
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        if process.stderr is not None and not process.stderr.closed:
            process.stderr.close()


def _prepare_mp4_frame(image: Image.Image) -> Image.Image:
    frame = image.convert("RGB")
    width, height = frame.size
    target_width = width + (width % 2)
    target_height = height + (height % 2)
    if (target_width, target_height) == frame.size:
        return frame

    padded = Image.new("RGB", (target_width, target_height), color="white")
    padded.paste(frame, (0, 0))
    return padded


def save_animation(images: Iterable[Image.Image], output_path: str | Path, fps: float, gif: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fps <= 0:
        raise ValueError(f"Animation FPS must be positive, got {fps}.")

    if gif:
        frames = [image.copy() for image in images]
        if not frames:
            raise ValueError("Cannot save an animation with no frames.")
        _save_gif(frames, output_path, fps)
    else:
        _save_mp4(images, output_path, fps)
    return output_path
