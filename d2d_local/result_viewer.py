"""Two-panel result window for a completed decode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from .pipeline import DecodeResult


@dataclass(frozen=True)
class ResultPanel:
    title: str
    path: Path | None
    row: int
    column: int
    unavailable_text: str
    status_text: str | None = None
    status_color: str | None = None


class ResultViewerError(RuntimeError):
    pass


def collect_result_panels(result: DecodeResult) -> tuple[ResultPanel, ...]:
    if not result.crc_ok:
        crc_text = "CRC FAILED - PREVIEW ONLY"
        crc_color = "#b71c1c"
    elif result.expected_sha256_ok is False:
        crc_text = "CRC PASSED - SHA-256 MISMATCH"
        crc_color = "#ef6c00"
    else:
        crc_text = "CRC PASSED"
        crc_color = "#2e7d32"
    return (
        ResultPanel(
            "Constellation",
            result.constellation_path,
            0,
            0,
            "Constellation not available",
        ),
        ResultPanel(
            "Recovered image",
            result.output_path,
            0,
            1,
            "Image recovery did not complete",
            crc_text,
            crc_color,
        ),
    )


def _preview(path: Path, size: tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    image.thumbnail(size, resampling)
    return image


def show_result_plots(result: DecodeResult, *, title: str = "OpenRadioPHY Results") -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
    except ImportError as exc:
        raise ResultViewerError("Tkinter is not available") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ResultViewerError(str(exc)) from exc

    root.withdraw()
    root.title(title)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = max(720, min(1500, int(screen_width * 0.9)))
    window_height = max(560, min(1000, int(screen_height * 0.88)))
    root.geometry(f"{window_width}x{window_height}")

    container = ttk.Frame(root, padding=8)
    container.pack(fill="both", expand=True)
    for index in (0, 1):
        container.columnconfigure(index, weight=1)
    container.rowconfigure(0, weight=1)

    panel_width = max(280, (window_width - 60) // 2)
    panel_height = max(360, window_height - 80)
    photos: list[ImageTk.PhotoImage] = []

    for panel in collect_result_panels(result):
        frame = ttk.LabelFrame(container, text=panel.title, padding=6)
        frame.grid(row=panel.row, column=panel.column, padx=5, pady=5, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        content_row = 0
        if panel.status_text is not None:
            tk.Label(
                frame,
                text=panel.status_text,
                background=panel.status_color,
                foreground="white",
                font=("TkDefaultFont", 10, "bold"),
                pady=4,
            ).grid(row=0, column=0, sticky="ew", pady=(0, 5))
            content_row = 1
        frame.rowconfigure(content_row, weight=1)

        if panel.path is None or not panel.path.is_file():
            ttk.Label(frame, text=panel.unavailable_text, anchor="center").grid(
                row=content_row,
                column=0,
                sticky="nsew",
            )
            continue

        try:
            preview = _preview(panel.path, (panel_width - 24, panel_height - 42))
        except (OSError, UnidentifiedImageError):
            ttk.Label(frame, text=f"Cannot preview {panel.path.name}", anchor="center").grid(
                row=content_row,
                column=0,
                sticky="nsew",
            )
            continue

        photo = ImageTk.PhotoImage(preview, master=root)
        photos.append(photo)
        ttk.Label(frame, image=photo, anchor="center").grid(
            row=content_row,
            column=0,
            sticky="nsew",
        )

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.deiconify()
    try:
        root.mainloop()
    except tk.TclError as exc:
        raise ResultViewerError(str(exc)) from exc
