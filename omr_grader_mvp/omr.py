from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from document_scanner import scan_document


PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
MARKER_SIZE = 70
MARKER_MARGIN = 75
MARKER_CENTERS = np.array(
    [
        [MARKER_MARGIN + MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
        [MARKER_MARGIN + MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
    ],
    dtype=np.float32,
)

QUESTION_COUNT = 20
CHOICES = ("A", "B", "C", "D")
BUBBLE_X = np.array([690, 900, 1110, 1320], dtype=np.int32)
QUESTION_START_Y = 430
QUESTION_GAP_Y = 78
BUBBLE_RADIUS = 28
INNER_RADIUS = 17


@dataclass(frozen=True)
class SheetLayout:
    question_count: int
    choices: tuple[str, ...]
    columns: int
    rows_per_column: int
    bubble_centers: tuple[tuple[tuple[int, int], ...], ...]
    bubble_radius: int
    inner_radius: int
    header: str = "OMR ANSWER SHEET"


def make_layout(question_count: int = 20, choices: Sequence[str] = ("A", "B", "C", "D", "E"), *, header: str = "OMR ANSWER SHEET") -> SheetLayout:
    if not 1 <= int(question_count) <= 100:
        raise ValueError("Question count must be between 1 and 100.")
    choices = tuple(str(c) for c in choices)
    if not 2 <= len(choices) <= 5:
        raise ValueError("Use between 2 and 5 answer choices.")

    # Auto-layout prioritizes scan reliability over squeezing bubbles too tightly.
    if question_count <= 30:
        columns = 1
    elif question_count <= 60:
        columns = 2
    elif question_count <= 90:
        columns = 3
    else:
        columns = 4
    rows_per_column = int(np.ceil(question_count / columns))

    left, right = 210, PAGE_WIDTH - 210
    top, bottom = 430, PAGE_HEIGHT - 190
    col_gap = 45
    col_width = (right - left - col_gap * (columns - 1)) / columns
    row_gap = min(78.0, (bottom - top) / max(rows_per_column - 1, 1))

    # Keep bubbles large enough for pencil marks and phone-camera capture.
    bubble_radius = int(np.clip(min(28.0, row_gap * 0.30, col_width / (len(choices) * 3.2)), 17, 28))
    inner_radius = max(10, int(round(bubble_radius * 0.61)))

    centers = []
    for q in range(question_count):
        col = q // rows_per_column
        row = q % rows_per_column
        x0 = left + col * (col_width + col_gap)
        qnum_space = max(48, bubble_radius * 2 + 12)
        choices_left = x0 + qnum_space
        choices_right = x0 + col_width - 12
        xs = np.linspace(choices_left + bubble_radius, choices_right - bubble_radius, len(choices))
        y = top + row * row_gap
        centers.append(tuple((int(round(x)), int(round(y))) for x in xs))

    return SheetLayout(
        question_count=int(question_count),
        choices=choices,
        columns=columns,
        rows_per_column=rows_per_column,
        bubble_centers=tuple(centers),
        bubble_radius=bubble_radius,
        inner_radius=inner_radius,
        header=header.strip() or "OMR ANSWER SHEET",
    )


DEFAULT_LAYOUT = make_layout(20, CHOICES)


@dataclass(frozen=True)
class QuestionDetection:
    question: int
    densities: tuple[float, ...]
    selected: str | None
    status: str
    confidence: float


@dataclass(frozen=True)
class AlignmentQuality:
    passed: bool
    score: float
    max_marker_error_px: float
    top_tilt_deg: float
    bottom_tilt_deg: float
    left_tilt_deg: float
    right_tilt_deg: float
    message: str


@dataclass(frozen=True)
class ScanResult:
    aligned_bgr: np.ndarray
    threshold: np.ndarray
    detections: tuple[QuestionDetection, ...]
    marker_detection_used: bool
    warning: str | None = None
    scanner_bgr: np.ndarray | None = None
    scanner_page_found: bool = False
    alignment_quality: AlignmentQuality | None = None
    layout: SheetLayout = DEFAULT_LAYOUT


class OMRProcessingError(RuntimeError):
    pass


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise OMRProcessingError("The uploaded file could not be decoded as an image.")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def _find_registration_markers(image_bgr: np.ndarray) -> np.ndarray | None:
    """Find the four solid black registration squares.

    Candidates are filtered for square shape / fill and then selected from four
    corner zones. This is intentionally stricter than choosing the four dark
    objects nearest the image corners, which could accidentally select text,
    bubble groups, or shadows.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    image_area = float(h * w)
    candidates: list[dict[str, float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.00022 or area > image_area * 0.025:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 8 or ch < 8:
            continue
        aspect = cw / float(ch)
        extent = area / float(cw * ch)
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = area / hull_area
        if not 0.72 <= aspect <= 1.28:
            continue
        if extent < 0.70 or solidity < 0.88:
            continue

        # Registration marks should be genuinely dark/solid, not just outlined.
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        pixels = gray[mask > 0]
        if pixels.size == 0:
            continue
        mean_gray = float(np.mean(pixels))
        dark_fraction = float(np.mean(pixels < 90))
        if mean_gray > 120 or dark_fraction < 0.62:
            continue

        candidates.append({
            "cx": x + cw / 2.0,
            "cy": y + ch / 2.0,
            "area": area,
            "extent": extent,
            "solidity": solidity,
            "dark_fraction": dark_fraction,
        })

    if len(candidates) < 4:
        return None

    targets = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    chosen: list[tuple[float, float]] = []
    used: set[int] = set()

    # Each real marker should live in the outer ~38% of its respective corner.
    for corner_index, target in enumerate(targets):
        scored: list[tuple[float, int]] = []
        for index, c in enumerate(candidates):
            if index in used:
                continue
            cx, cy = c["cx"], c["cy"]
            if corner_index == 0 and not (cx < w * 0.42 and cy < h * 0.42):
                continue
            if corner_index == 1 and not (cx > w * 0.58 and cy < h * 0.42):
                continue
            if corner_index == 2 and not (cx > w * 0.58 and cy > h * 0.58):
                continue
            if corner_index == 3 and not (cx < w * 0.42 and cy > h * 0.58):
                continue

            dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - target)) / max(w, h)
            shape_bonus = 0.05 * c["extent"] + 0.05 * c["solidity"] + 0.06 * c["dark_fraction"]
            area_bonus = min(0.05, np.sqrt(c["area"]) / max(w, h) * 0.5)
            scored.append((dist - shape_bonus - area_bonus, index))

        if not scored:
            return None
        _, best_index = min(scored)
        used.add(best_index)
        c = candidates[best_index]
        chosen.append((c["cx"], c["cy"]))

    ordered = _order_points(np.array(chosen, dtype=np.float32))
    tl, tr, br, bl = ordered
    horizontal = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    vertical = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if horizontal < w * 0.50 or vertical < h * 0.55:
        return None

    return ordered


def _edge_tilt_degrees(a: np.ndarray, b: np.ndarray, *, vertical: bool = False) -> float:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if vertical:
        # Perfect vertical is +/-90 degrees. Report deviation from vertical.
        return min(abs(angle - 90.0), abs(angle + 90.0))
    return min(abs(angle), abs(abs(angle) - 180.0))


def _alignment_quality(markers: np.ndarray | None) -> AlignmentQuality:
    if markers is None:
        return AlignmentQuality(
            False, 0.0, 999.0, 99.0, 99.0, 99.0, 99.0,
            "Registration-square alignment could not be verified. Retake the photo with all four black squares clearly visible."
        )

    ordered = _order_points(markers)
    tl, tr, br, bl = ordered
    errors = np.linalg.norm(ordered - MARKER_CENTERS, axis=1)
    max_error = float(np.max(errors))
    top = _edge_tilt_degrees(tl, tr)
    bottom = _edge_tilt_degrees(bl, br)
    left = _edge_tilt_degrees(tl, bl, vertical=True)
    right = _edge_tilt_degrees(tr, br, vertical=True)
    max_tilt = max(top, bottom, left, right)

    error_score = np.clip(1.0 - max_error / 35.0, 0.0, 1.0)
    tilt_score = np.clip(1.0 - max_tilt / 2.0, 0.0, 1.0)
    score = float(100.0 * (0.65 * error_score + 0.35 * tilt_score))
    passed = max_error <= 24.0 and max_tilt <= 1.5
    if passed:
        message = f"Fine registration-square alignment passed ({score:.0f}/100)."
    else:
        message = (
            f"Fine alignment is uncertain (marker error {max_error:.1f}px, max tilt {max_tilt:.2f}°). "
            "Retake the photo with the whole sheet flatter and all four black squares visible."
        )
    return AlignmentQuality(passed, score, max_error, top, bottom, left, right, message)

def _fallback_page_warp(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            continue
        source = _order_points(approx.reshape(4, 2))
        destination = np.array(
            [[0, 0], [PAGE_WIDTH - 1, 0], [PAGE_WIDTH - 1, PAGE_HEIGHT - 1], [0, PAGE_HEIGHT - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(image_bgr, matrix, (PAGE_WIDTH, PAGE_HEIGHT)), (
            "Registration squares were not found; the app used the page boundary instead. "
            "For the most reliable grading, keep all four black corner squares visible."
        )

    resized = cv2.resize(image_bgr, (PAGE_WIDTH, PAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized, (
        "The app could not detect the corner squares or page boundary, so it resized the image directly. "
        "Results may be inaccurate; retake the photo with the full sheet visible."
    )


def align_sheet(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str | None, AlignmentQuality]:
    """Fine-align the OMR sheet using its four registration squares.

    A first marker warp removes most residual perspective. We then detect the
    markers again on the normalized sheet and, when useful, apply a second tiny
    corrective warp. This prevents a page that looks "almost straight" from
    leaving the bubble grid slanted.
    """
    markers = _find_registration_markers(image_bgr)
    if markers is None:
        aligned, warning = _fallback_page_warp(image_bgr)
        quality = _alignment_quality(_find_registration_markers(aligned))
        warning = warning + " " + quality.message
        return aligned, False, warning, quality

    destination = MARKER_CENTERS.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(markers.astype(np.float32), destination)
    aligned = cv2.warpPerspective(
        image_bgr,
        matrix,
        (PAGE_WIDTH, PAGE_HEIGHT),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Second pass: use the markers again after the first normalization. Any
    # remaining slant / trapezoid error is corrected before bubbles are read.
    residual_markers = _find_registration_markers(aligned)
    if residual_markers is not None:
        residual_error = float(np.max(np.linalg.norm(residual_markers - destination, axis=1)))
        if residual_error > 2.0:
            correction = cv2.getPerspectiveTransform(residual_markers.astype(np.float32), destination)
            aligned = cv2.warpPerspective(
                aligned,
                correction,
                (PAGE_WIDTH, PAGE_HEIGHT),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

    final_markers = _find_registration_markers(aligned)
    quality = _alignment_quality(final_markers)
    warning = None if quality.passed else quality.message
    return aligned, True, warning, quality

def _prepare_threshold(aligned_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold handles mild shadows and inexpensive phone cameras better than a global threshold.
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12,
    )
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return threshold


def bubble_centers(layout: SheetLayout = DEFAULT_LAYOUT) -> list[list[tuple[int, int]]]:
    return [list(row) for row in layout.bubble_centers]


def _circle_mask(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _bubble_fill_score(gray: np.ndarray, center: tuple[int, int], radius: int = INNER_RADIUS) -> float:
    """Score a bubble using only its inner area and its local paper background.

    A score near 0 means the center looks like surrounding paper. Higher scores mean
    the center is substantially darker. The printed bubble ring is intentionally
    excluded so it cannot make an empty bubble look filled.
    """
    h, w = gray.shape
    cx, cy = center
    if cx < 52 or cy < 52 or cx >= w - 52 or cy >= h - 52:
        return 0.0

    inner_mask = _circle_mask(gray.shape, center, radius)
    outer = _circle_mask(gray.shape, center, 47)
    exclude_ring = _circle_mask(gray.shape, center, 34)
    background_mask = cv2.subtract(outer, exclude_ring)

    inner_pixels = gray[inner_mask > 0].astype(np.float32)
    background_pixels = gray[background_mask > 0].astype(np.float32)
    if inner_pixels.size == 0 or background_pixels.size == 0:
        return 0.0

    # Robust local white level. Using a high percentile handles small text or
    # accidental dark pixels in the annulus better than a plain average.
    local_background = float(np.percentile(background_pixels, 72))

    # Two complementary signals: how much of the center is clearly dark, and
    # how much darker its average is than its own local paper background.
    dark_cut = max(0.0, local_background - 24.0)
    dark_fraction = float(np.mean(inner_pixels < dark_cut))
    mean_darkness = max(0.0, (local_background - float(np.mean(inner_pixels))) / 170.0)
    lower_quartile_darkness = max(
        0.0,
        (local_background - float(np.percentile(inner_pixels, 35))) / 190.0,
    )

    score = 0.58 * dark_fraction + 0.27 * mean_darkness + 0.15 * lower_quartile_darkness
    return float(np.clip(score, 0.0, 1.0))

def classify_densities(
    densities: Sequence[float],
    *,
    answer_key_mode: bool,
    choices: Sequence[str] = CHOICES,
) -> tuple[str | None, str, float]:
    """Choose the strongest bubble when it clearly dominates the alternatives.

    The absolute score prevents random noise from becoming an answer. The relative
    margin/ratio lets a clearly dominant bubble win even when lighting or pencil
    pressure makes all scores lower than usual. Ambiguous rows are sent to review.
    """
    values = np.asarray(densities, dtype=np.float32)
    order = np.argsort(values)[::-1]
    top_index = int(order[0])
    second_index = int(order[1])
    top = float(values[top_index])
    second = float(values[second_index])
    margin = top - second
    ratio = top / max(second, 0.025)

    # Answer keys should normally be strongly filled, so they can be slightly stricter.
    blank_floor = 0.115 if answer_key_mode else 0.085
    useful_mark = 0.18 if answer_key_mode else 0.135
    strong_mark = 0.30 if answer_key_mode else 0.235

    # Confidence is a practical separation indicator, not a probability.
    margin_conf = np.clip(margin / 0.22, 0.0, 1.0)
    ratio_conf = np.clip((ratio - 1.0) / 1.5, 0.0, 1.0)
    strength_conf = np.clip((top - blank_floor) / max(0.01, strong_mark - blank_floor), 0.0, 1.0)
    confidence = float(100.0 * (0.45 * margin_conf + 0.35 * ratio_conf + 0.20 * strength_conf))

    if top < blank_floor:
        return None, "blank", 0.0

    # Clearly dominant winner. This is the main new behavior requested: the
    # darkest/most-filled bubble wins when it separates meaningfully from #2.
    clear_by_margin = margin >= (0.075 if answer_key_mode else 0.055)
    clear_by_ratio = ratio >= (1.48 if answer_key_mode else 1.38)
    very_strong = top >= strong_mark and margin >= 0.040
    if top >= useful_mark and ((clear_by_margin and clear_by_ratio) or very_strong):
        return str(choices[top_index]), "clear", confidence

    # Two substantial, similar marks are more likely a double mark/erase conflict.
    if second >= useful_mark * 0.90 and (ratio < 1.32 or margin < 0.050):
        return None, "multiple", confidence

    # There is some marking, but not enough separation to safely guess.
    return None, "unclear", confidence

def scan_sheet(
    image_bgr: np.ndarray,
    *,
    answer_key_mode: bool,
    scanner_mode: bool = False,
    layout: SheetLayout = DEFAULT_LAYOUT,
) -> ScanResult:
    scanner_bgr = None
    scanner_page_found = False
    scanner_warning = None

    working_image = image_bgr
    if scanner_mode:
        scanned = scan_document(image_bgr, output_size=(PAGE_WIDTH, PAGE_HEIGHT))
        scanner_bgr = scanned.image_bgr
        scanner_page_found = scanned.page_found
        # Analyze the detail-preserving scan. The more aggressive white version is
        # only the preview, so faint pencil marks are not accidentally erased.
        working_image = scanned.analysis_bgr
        if not scanned.page_found:
            scanner_warning = scanned.message

    aligned, marker_used, warning, alignment_quality = align_sheet(working_image)
    if scanner_warning and warning:
        warning = scanner_warning + " " + warning
    elif scanner_warning:
        warning = scanner_warning
    threshold = _prepare_threshold(aligned)
    analysis_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    detections: list[QuestionDetection] = []
    for q_index, row in enumerate(bubble_centers(layout), start=1):
        densities = tuple(_bubble_fill_score(analysis_gray, center, radius=layout.inner_radius) for center in row)
        selected, status, confidence = classify_densities(densities, answer_key_mode=answer_key_mode, choices=layout.choices)
        detections.append(
            QuestionDetection(
                question=q_index,
                densities=densities,
                selected=selected,
                status=status,
                confidence=confidence,
            )
        )

    return ScanResult(
        aligned_bgr=aligned,
        threshold=threshold,
        detections=tuple(detections),
        marker_detection_used=marker_used,
        warning=warning,
        scanner_bgr=scanner_bgr,
        scanner_page_found=scanner_page_found,
        alignment_quality=alignment_quality,
        layout=layout,
    )


def crop_question(aligned_bgr: np.ndarray, question_number: int, layout: SheetLayout = DEFAULT_LAYOUT) -> np.ndarray:
    if not 1 <= question_number <= layout.question_count:
        raise ValueError("Question number is outside the current sheet layout.")
    row = layout.bubble_centers[question_number - 1]
    xs = [p[0] for p in row]
    y = row[0][1]
    pad_x = max(55, layout.bubble_radius * 2)
    pad_y = max(36, layout.bubble_radius + 12)
    x1 = max(0, min(xs) - pad_x - 55)
    x2 = min(PAGE_WIDTH, max(xs) + pad_x)
    y1 = max(0, y - pad_y)
    y2 = min(PAGE_HEIGHT, y + pad_y)
    return aligned_bgr[y1:y2, x1:x2].copy()


def draw_overlay(
    aligned_bgr: np.ndarray,
    detections: Sequence[QuestionDetection],
    *,
    key_answers: Sequence[str] | None = None,
    overrides: dict[int, str | None] | None = None,
    layout: SheetLayout = DEFAULT_LAYOUT,
) -> np.ndarray:
    overlay = aligned_bgr.copy()
    overrides = overrides or {}

    for detection in detections:
        q = detection.question
        row = layout.bubble_centers[q - 1]
        xs = [p[0] for p in row]
        y = row[0][1]
        pad = max(12, layout.bubble_radius // 2)
        x1, x2 = max(0, min(xs) - 75), min(PAGE_WIDTH - 1, max(xs) + 55)
        y1, y2 = max(0, y - layout.bubble_radius - pad), min(PAGE_HEIGHT - 1, y + layout.bubble_radius + pad)

        selected = overrides.get(q, detection.selected)
        if selected == "Blank":
            selected = None

        if key_answers is None:
            if detection.status == "clear" or q in overrides:
                color = (50, 170, 70)  # green
            elif detection.status == "blank":
                color = (145, 145, 145)  # gray
            else:
                color = (0, 165, 255)  # orange
        else:
            if detection.status not in {"clear"} and q not in overrides:
                color = (0, 165, 255)
            elif selected is None:
                color = (145, 145, 145)
            elif selected == key_answers[q - 1]:
                color = (50, 170, 70)
            else:
                color = (45, 45, 220)  # red in BGR

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
        label = selected if selected is not None else detection.status.upper()
        cv2.putText(
            overlay,
            f"Q{q}: {label}",
            (min(PAGE_WIDTH - 180, x2 + 12), y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def answers_from_detections(
    detections: Sequence[QuestionDetection],
    overrides: dict[int, str | None] | None = None,
) -> list[str | None]:
    overrides = overrides or {}
    answers: list[str | None] = []
    for detection in detections:
        value = overrides.get(detection.question, detection.selected)
        if value == "Blank":
            value = None
        answers.append(value)
    return answers


def grade_answers(student_answers: Sequence[str | None], key_answers: Sequence[str]) -> dict[str, object]:
    if len(student_answers) != len(key_answers):
        raise ValueError("Student answers and answer key must have the same number of questions.")

    rows: list[dict[str, object]] = []
    correct = 0
    blank = 0
    for index, (student, correct_answer) in enumerate(zip(student_answers, key_answers), start=1):
        is_correct = student == correct_answer
        if is_correct:
            correct += 1
        if student is None:
            blank += 1
        rows.append(
            {
                "Question": index,
                "Student answer": student or "Blank",
                "Correct answer": correct_answer,
                "Result": "Correct" if is_correct else ("Blank" if student is None else "Incorrect"),
            }
        )

    total = len(key_answers)
    return {
        "score": correct,
        "total": total,
        "percentage": (correct / total * 100.0) if total else 0.0,
        "blank": blank,
        "incorrect": total - correct - blank,
        "rows": rows,
    }


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise OMRProcessingError("Could not encode the processed image.")
    return encoded.tobytes()


def detection_rows(detections: Iterable[QuestionDetection], choices: Sequence[str] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detection in detections:
        row = {
            "Question": detection.question,
            "Detected": detection.selected or detection.status.title(),
            "Status": detection.status.title(),
            "Confidence": round(detection.confidence, 1),
        }
        labels = list(choices) if choices is not None else [chr(ord("A") + i) for i in range(len(detection.densities))]
        for label, density in zip(labels, detection.densities):
            row[f"{label} density"] = round(density, 3)
        rows.append(row)
    return rows
