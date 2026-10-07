from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omr import MARKER_MARGIN, MARKER_SIZE, PAGE_HEIGHT, PAGE_WIDTH, SheetLayout, make_layout


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def create_sheet(
    answer_marks: Sequence[str | None] | None = None,
    *,
    title: str = "OMR ANSWER SHEET",
    question_count: int = 20,
    choices: Sequence[str] = ("A", "B", "C", "D"),
    layout: SheetLayout | None = None,
) -> np.ndarray:
    """Create a pure-white, scanner-safe answer sheet.

    The teacher can change only the header and number of questions. The layout
    engine controls bubble geometry so custom sheets remain reliable to scan.
    """
    layout = layout or make_layout(question_count, choices, header=title)
    if answer_marks is not None and len(answer_marks) != layout.question_count:
        raise ValueError(f"Expected {layout.question_count} answers.")

    # Important: no alternating gray rows or tinted background. The page stays
    # exactly white so phone scans and thresholding have a clean baseline.
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    marker_positions = [
        (MARKER_MARGIN, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
        (MARKER_MARGIN, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
    ]
    for x, y in marker_positions:
        draw.rectangle((x, y, x + MARKER_SIZE, y + MARKER_SIZE), fill="black")

    # Header and name are intentionally the only editable/identity fields.
    safe_title = (title or "OMR ANSWER SHEET").strip()[:70]
    title_size = 52 if len(safe_title) <= 35 else 42 if len(safe_title) <= 50 else 34
    draw.text((PAGE_WIDTH // 2, 112), safe_title, font=_font(title_size, bold=True), fill="black", anchor="ma")
    draw.text(
        (PAGE_WIDTH // 2, 175),
        "Fill one bubble completely for each question. Keep all four black squares visible when scanning.",
        font=_font(22),
        fill="black",
        anchor="ma",
    )
    draw.line((205, 240, PAGE_WIDTH - 205, 240), fill="black", width=3)
    draw.text((240, 285), "Name: ________________________________________________", font=_font(28), fill="black")

    # Column headers and questions. No gray shading is ever drawn.
    for col in range(layout.columns):
        start_q = col * layout.rows_per_column
        if start_q >= layout.question_count:
            break
        first_row = layout.bubble_centers[start_q]
        label_y = first_row[0][1] - max(58, layout.bubble_radius * 2 + 14)
        qx = first_row[0][0] - max(80, layout.bubble_radius * 3)
        draw.text((qx, label_y), "Q", font=_font(24, bold=True), fill="black", anchor="mm")
        for center, choice in zip(first_row, layout.choices):
            draw.text((center[0], label_y), str(choice), font=_font(24, bold=True), fill="black", anchor="mm")

    q_font = max(18, min(27, int(layout.bubble_radius * 0.95)))
    outline_width = max(3, int(round(layout.bubble_radius * 0.14)))
    for q_index, row in enumerate(layout.bubble_centers):
        q = q_index + 1
        y = row[0][1]
        qx = row[0][0] - max(80, layout.bubble_radius * 3)
        draw.text((qx, y), str(q), font=_font(q_font, bold=True), fill="black", anchor="mm")
        for choice_index, (x, by) in enumerate(row):
            r = layout.bubble_radius
            box = (x - r, by - r, x + r, by + r)
            draw.ellipse(box, outline="black", width=outline_width)
            if answer_marks is not None and answer_marks[q_index] == layout.choices[choice_index]:
                inner = max(5, r - max(6, r // 4))
                draw.ellipse((x-inner, by-inner, x+inner, by+inner), fill="black")

    footer = f"{layout.question_count} questions · choices {'-'.join((layout.choices[0], layout.choices[-1]))} · {layout.columns} column{'s' if layout.columns != 1 else ''}"
    draw.text((PAGE_WIDTH // 2, PAGE_HEIGHT - 112), footer, font=_font(20), fill="black", anchor="ma")

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_sheet(path: str | Path, **kwargs) -> None:
    image = create_sheet(**kwargs)
    cv2.imwrite(str(path), image)


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "assets" / "omr_answer_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_sheet(output, question_count=30, choices=("A", "B", "C", "D", "E"))
    print(output)
