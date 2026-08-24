"""Visual-log rendering for ProcTHOR evaluations: annotated frames saved as MP4."""
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.utils.fonts import load_font


def _ellipsis_fit(font, text, max_width):
    if not text:
        return text
    ell = "..."
    w = font.getbbox(text)[2] - font.getbbox(text)[0]
    if w <= max_width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ell
        w = font.getbbox(cand)[2] - font.getbbox(cand)[0]
        if w <= max_width:
            lo = mid + 1
        else:
            hi = mid
    return text[:max(lo - 1, 0)].rstrip() + ell


def _truncate_lines_with_ellipsis(lines, max_lines, font, max_width):
    if max_lines is None or len(lines) <= max_lines:
        return lines
    trimmed = lines[:max_lines]
    trimmed[-1] = _ellipsis_fit(font, trimmed[-1], max_width)
    trimmed.append("...")
    return trimmed


def _wrap_lines(font, text, max_width):
    """Wrap text to max_width, preserving explicit newlines."""
    lines = []
    for paragraph in str(text).split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            bbox = font.getbbox(test)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    return lines


def _line_size(font, s):
    bbox = font.getbbox(s if s else " ")
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def add_text_overlay(
    img: Image.Image,
    text: str,
    font_path="UbuntuMono-B.ttf",
    action_font_size=32,
    padding=16,
    bg_color=(0, 0, 0, 150),
    text_fill=(255, 255, 255, 255),
    line_gap=6,
) -> Image.Image:
    """Draw text over the top of the image on a translucent band."""
    im = img.convert("RGBA")
    font = load_font(font_path, action_font_size)

    usable_width = im.width - 2 * padding
    lines = _wrap_lines(font, text, usable_width)

    total_h = 0
    heights = []
    for ln in lines:
        _, lh = _line_size(font, ln)
        heights.append(lh)
        total_h += lh + line_gap
    if lines:
        total_h -= line_gap

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    top = padding - 6
    bottom = min(im.height - padding + 6, top + total_h + 12)
    o.rectangle([0, top, im.width, bottom], fill=bg_color)

    im = Image.alpha_composite(im, overlay)
    draw = ImageDraw.Draw(im)

    y = padding
    for ln, lh in zip(lines, heights):
        if y + lh > im.height - padding:
            draw.text((padding, y), "...", font=font, fill=text_fill)
            break
        draw.text((padding, y), ln, font=font, fill=text_fill)
        y += lh + line_gap

    return im.convert("RGB")


def compose_left_text_right_image(
    img: Image.Image,
    text: str,
    font_path="UbuntuMono-B.ttf",
    action_font_size=32,
    padding=16,
    panel_bg=(0, 0, 0, 150),
    text_fill=(255, 255, 255, 255),
    line_gap=6,
) -> Image.Image:
    """Text panel on the left, original image on the right."""
    base = img.convert("RGBA")
    W, H = base.size
    out = Image.new("RGBA", (W * 2, H), (255, 255, 255, 0))

    panel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    p = ImageDraw.Draw(panel)
    p.rectangle([0, 0, W, H], fill=panel_bg)

    font = load_font(font_path, action_font_size)
    max_text_w = W - 2 * padding
    lines = _wrap_lines(font, text, max_text_w)

    y = padding
    for ln in lines:
        _, lh = _line_size(font, ln)
        if y + lh > H - padding:
            p.text((padding, y), "...", font=font, fill=text_fill)
            break
        p.text((padding, y), ln, font=font, fill=text_fill)
        y += lh + line_gap

    out.alpha_composite(panel, (0, 0))
    out.alpha_composite(base, (W, 0))
    return out.convert("RGB")


def compose_overlay_plus_right_text_dynamic_height(
    img: Image.Image,
    overlay_text: str,
    right_text: str,
    font_path="UbuntuMono-B.ttf",
    action_font_size=32,
    right_text_font_size=32,
    padding=16,
    panel_bg=(0, 0, 0, 150),
    text_fill=(255, 255, 255, 255),
    line_gap=6,
    max_right_lines: Optional[int] = 10,
) -> Image.Image:
    """Left: image with a translucent text overlay. Right: text panel.

    The canvas height grows to fit the right-hand text.
    """
    W, H = img.size

    font_right = load_font(font_path, right_text_font_size)
    max_text_w = W - 2 * padding
    right_lines = _wrap_lines(font_right, right_text, max_text_w)
    right_lines = _truncate_lines_with_ellipsis(right_lines, max_right_lines, font_right, max_text_w)
    total_text_h = 0
    line_heights = []
    for ln in right_lines:
        _, lh = _line_size(font_right, ln)
        line_heights.append(lh)
        total_text_h += lh + line_gap
    if right_lines:
        total_text_h -= line_gap

    needed_height = max(H, total_text_h + 2 * padding)

    left_canvas = Image.new("RGBA", (W, needed_height), (0, 0, 0, 0))
    left_canvas.paste(img.convert("RGBA"), (0, 0))

    left_with_overlay = add_text_overlay(
        left_canvas.convert("RGB"),
        overlay_text,
        font_path=font_path,
        action_font_size=action_font_size,
        padding=padding,
        bg_color=panel_bg,
        text_fill=text_fill,
        line_gap=line_gap,
    ).convert("RGBA")

    right_panel = Image.new("RGBA", (W, needed_height), (0, 0, 0, 0))
    rp = ImageDraw.Draw(right_panel)
    rp.rectangle([0, 0, W, needed_height], fill=panel_bg)

    y = padding
    for ln, lh in zip(right_lines, line_heights):
        rp.text((padding, y), ln, font=font_right, fill=text_fill)
        y += lh + line_gap

    out = Image.new("RGBA", (W * 2, needed_height), (255, 255, 255, 0))
    out.alpha_composite(left_with_overlay, (0, 0))
    out.alpha_composite(right_panel, (W, 0))
    return out.convert("RGB")


def render_by_vis_type(
    img: Image.Image,
    text_main: str,
    vis_type: str,
    text_right: str = "",
    font_path="UbuntuMono-B.ttf",
    action_font_size=32,
    right_text_font_size=32,
    max_right_lines=6,
) -> Image.Image:
    if vis_type == "dec_only":
        return add_text_overlay(img, text_main, font_path=font_path, action_font_size=action_font_size)
    elif vis_type == "dec_split":
        return compose_left_text_right_image(img, text_main, font_path=font_path, action_font_size=action_font_size)
    elif vis_type == "dec_obs":
        return compose_overlay_plus_right_text_dynamic_height(
            img, text_main, text_right, font_path=font_path,
            action_font_size=action_font_size, right_text_font_size=right_text_font_size,
            max_right_lines=max_right_lines,
        )
    else:
        return add_text_overlay(img, text_main, font_path=font_path, action_font_size=action_font_size)


def _pad_to_same_size(imgs: Sequence[Image.Image], bg_color="white") -> List[Image.Image]:
    """Pad variable-size frames onto identical canvases (top-left aligned)."""
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    padded = []
    for im in imgs:
        canvas = Image.new("RGB", (max_w, max_h), bg_color)
        canvas.paste(im, (0, 0))
        padded.append(canvas)
    return padded


def save_images_as_mp4(
    img_list: Sequence[Image.Image],
    text_list: Sequence[str],
    right_text_list: Optional[Sequence[str]],
    file_name: str,
    vis_type: str = "dec_obs",
    font_path: str = "UbuntuMono-B.ttf",
    action_font_size: int = 32,
    right_text_font_size: int = 32,
    bg_color: str = "white",
    duration_ms: int = 500,
    loop: int = 0,
    max_right_lines: int = 6,
    codec: str = "mp4v",
):
    """Render annotated frames and save them as an MP4.

    duration_ms sets the per-frame display time (fps = 1000 / duration_ms).
    loop > 0 repeats the frame sequence (loop + 1) times.
    """
    if right_text_list is None:
        right_text_list = [""] * len(img_list)
    assert len(img_list) == len(text_list) == len(right_text_list), "List lengths must match."

    processed = [
        render_by_vis_type(
            img, text_main, vis_type=vis_type, text_right=right_txt,
            font_path=font_path, action_font_size=action_font_size,
            right_text_font_size=right_text_font_size, max_right_lines=max_right_lines,
        )
        for img, text_main, right_txt in zip(img_list, text_list, right_text_list)
    ]

    frames = _pad_to_same_size(processed, bg_color=bg_color)
    if not frames:
        raise ValueError("No frames to save.")

    fps = max(1, round(1000 / duration_ms))
    h, w = frames[0].height, frames[0].width

    repeat = (loop + 1) if loop >= 0 else 1
    all_frames = frames * repeat

    root, ext = os.path.splitext(file_name)
    if ext.lower() != ".mp4":
        file_name = root + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(file_name, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for '{file_name}' with codec '{codec}'.")

    try:
        for pil_im in all_frames:
            frame = np.array(pil_im.convert("RGB"))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
