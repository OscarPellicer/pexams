import logging

import cv2
import numpy as np
import pytest

from pexams.correct_exams import _find_fiducial_markers


@pytest.fixture(autouse=True)
def debug_logging():
    previous_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        yield
    finally:
        logging.getLogger().setLevel(previous_level)


def _draw_cross(image, center, arm=65, thickness=7):
    x, y = center
    half_arm = arm // 2
    half_thickness = thickness // 2
    cv2.rectangle(image, (x - half_thickness, y - half_arm), (x + half_thickness, y + half_arm), (0, 0, 0), -1)
    cv2.rectangle(image, (x - half_arm, y - half_thickness), (x + half_arm, y + half_thickness), (0, 0, 0), -1)


def _make_synthetic_scan(corners, add_false_positive_text=False):
    image = np.full((2339, 1654, 3), 255, dtype=np.uint8)
    for corner in corners:
        _draw_cross(image, tuple(map(int, corner)))

    # Add layout-like content and filled answer boxes so the image resembles
    # a real sheet, without storing any student data in the repository.
    cv2.putText(image, "Synthetic Exam - Regression Sheet", (470, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(image, "Course: Synthetic", (180, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    cv2.rectangle(image, (180, 540), (560, 610), (0, 0, 0), 2)
    cv2.rectangle(image, (180, 690), (740, 775), (0, 0, 0), 2)
    cv2.rectangle(image, (810, 690), (1370, 775), (0, 0, 0), 2)

    for row in range(12):
        y = 900 + row * 40
        cv2.putText(image, str(row + 1), (180, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        for col in range(4):
            x = 255 + col * 55
            cv2.rectangle(image, (x, y), (x + 38, y + 15), (0, 0, 0), 2)
        cv2.rectangle(image, (255 + (row % 4) * 55, y), (293 + (row % 4) * 55, y + 15), (90, 90, 90), -1)

    if add_false_positive_text:
        # This mimics the old failure where a letter in the top-right corner
        # bucket looked cross-like enough to be selected instead of the marker.
        cv2.putText(image, "a", (1335, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

    return image


def _rotate_image(image, degrees):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return rotated, matrix


def _adjust_brightness_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _transform_points(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))]
    return homogeneous @ matrix.T


@pytest.mark.parametrize("add_false_positive_text", [False, True])
def test_fiducials_are_detected_on_synthetic_problem_scans(add_false_positive_text, tmp_path):
    expected = np.array(
        [
            [175, 216],
            [1455, 214],
            [1454, 2115],
            [174, 2117],
        ],
        dtype=np.float32,
    )
    image = _make_synthetic_scan(expected, add_false_positive_text=add_false_positive_text)
    page_name = f"synthetic_false_positive_{add_false_positive_text}"

    corners = _find_fiducial_markers(image, debug_dir=str(tmp_path), page_number=page_name)

    assert corners is not None
    np.testing.assert_allclose(corners, expected, atol=10)
    assert (tmp_path / f"page_{page_name}_1_thresh.png").exists()
    assert (tmp_path / f"page_{page_name}_2_candidates.png").exists()
    assert (tmp_path / f"page_{page_name}_3_centroids.png").exists()


@pytest.mark.parametrize("degrees", [-1.0, -0.5, 0.5, 1.0])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0), (0.75, 20), (1.25, -20)])
def test_fiducials_survive_small_rotation_and_contrast_changes(degrees, alpha, beta, tmp_path):
    expected = np.array(
        [
            [175, 216],
            [1455, 214],
            [1454, 2115],
            [174, 2117],
        ],
        dtype=np.float32,
    )
    image = _make_synthetic_scan(expected, add_false_positive_text=True)

    rotated, matrix = _rotate_image(image, degrees)
    adjusted = _adjust_brightness_contrast(rotated, alpha, beta)

    corners = _find_fiducial_markers(
        adjusted,
        debug_dir=str(tmp_path),
        page_number=f"jocarchi_rot_{degrees}_alpha_{alpha}_beta_{beta}",
    )

    assert corners is not None
    np.testing.assert_allclose(corners, _transform_points(expected, matrix), atol=25)
