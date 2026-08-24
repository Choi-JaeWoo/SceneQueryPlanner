from PIL import ImageFont

_FALLBACK_FONTS = [
    "UbuntuMono-B.ttf",
    "DejaVuSansMono-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]


def load_font(font_path, size):
    """Load a truetype font, falling back to common system fonts or PIL's default."""
    for candidate in [font_path] + _FALLBACK_FONTS:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()
