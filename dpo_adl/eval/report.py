from __future__ import annotations

import glob
from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def bundle_plots_to_pdf(plots_dir: Path, out_pdf: Path, order: List[str] | None = None) -> None:
    plots_dir = Path(plots_dir)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Discover plot images
    imgs = sorted([Path(p) for p in glob.glob(str(plots_dir / "*.png"))])
    if order:
        # Reorder by listed prefixes, append remaining at end
        ordered = []
        for prefix in order:
            for p in imgs:
                if p.name.startswith(prefix):
                    ordered.append(p)
        remaining = [p for p in imgs if p not in ordered]
        imgs = ordered + remaining

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter
    margins = 36  # 0.5 inch
    avail_w = width - 2 * margins
    avail_h = height - 2 * margins

    for img_path in imgs:
        try:
            img = ImageReader(str(img_path))
            iw, ih = img.getSize()
            # Scale to fit preserving aspect
            scale = min(avail_w / iw, avail_h / ih)
            sw, sh = iw * scale, ih * scale
            x = (width - sw) / 2
            y = (height - sh) / 2
            c.drawImage(img, x, y, width=sw, height=sh)
            # Caption
            c.setFont("Helvetica", 10)
            c.drawString(margins, margins / 2, img_path.name)
            c.showPage()
        except Exception:
            # Skip problematic image
            continue
    c.save()

