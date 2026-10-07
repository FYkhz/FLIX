from __future__ import annotations

import cv2
import numpy as np

from omr import answers_from_detections, make_layout, scan_sheet
from sheet_template import create_sheet


def perspective_photo(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst = np.float32([[250, 210], [1910, 110], [1810, 2410], [160, 2290]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, m, (2200, 2600), borderValue=(185,185,185))


def run_case(n: int):
    layout = make_layout(n, tuple("ABCDE"), header=f"TEST {n}")
    answers = ["ABCDE"[i % 5] for i in range(n)]
    img = create_sheet(answers, title=f"TEST {n}", layout=layout, question_count=n, choices=layout.choices)
    photo = perspective_photo(img)
    scan = scan_sheet(photo, answer_key_mode=True, layout=layout)
    got = answers_from_detections(scan.detections)
    mismatches = [(i+1,a,b) for i,(a,b) in enumerate(zip(answers,got)) if a != b]
    print(n, "columns", layout.columns, "radius", layout.bubble_radius, "alignment", scan.alignment_quality.score if scan.alignment_quality else None, "mismatches", len(mismatches))
    if mismatches:
        print(mismatches[:10])
    assert got == answers


if __name__ == "__main__":
    for n in (30, 90, 100):
        run_case(n)
    print("Custom layout tests passed.")
