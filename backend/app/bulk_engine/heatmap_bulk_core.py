import csv
import io
import pickle
from typing import Sequence, Union

import cv2
import numpy as np


def _find_minimum_distance_point(
    coefficients: np.ndarray,
    x_q: float,
    y_q: float,
    min_x: float,
    max_x: float,
    *,
    poly: np.poly1d | None = None,
    poly_der: np.poly1d | None = None,
) -> tuple[float, tuple[float, float]]:
    if poly is None or poly_der is None:
        poly = np.poly1d(coefficients)
        poly_der = np.polyder(poly)
    g_prime = 2 * np.poly1d([1, -x_q]) + 2 * (poly - y_q) * poly_der

    candidates = [x_q]
    if np.isfinite(min_x):
        candidates.append(min_x)
    if np.isfinite(max_x):
        candidates.append(max_x)

    try:
        roots = np.roots(g_prime)
        for root in roots:
            if np.isreal(root):
                x_val = float(np.real(root))
                if min_x <= x_val <= max_x:
                    candidates.append(x_val)
    except Exception:
        pass

    def distance_sq(x_val: float) -> float:
        return (x_val - x_q) ** 2 + (poly(x_val) - y_q) ** 2

    best_x = min(candidates, key=distance_sq)
    min_distance = float(np.sqrt(distance_sq(best_x)))
    min_point = (float(best_x), float(poly(best_x)))
    return min_distance, min_point


def _poly_fit(values: Sequence[Sequence[float]], degree: int = 1) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size == 0:
        return np.array([])
    u1_values = values_arr[:, 1]
    f_values = values_arr[:, 0]
    W = np.vander(u1_values, degree + 1)
    try:
        coefficients = np.linalg.inv(W.T @ W) @ W.T @ f_values
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(W) @ f_values
    return coefficients


def _basis_conversion(
    contour: Sequence[Sequence[int]],
    X: np.ndarray,
    center_x: float,
    center_y: float,
    coordinates_inside_cell: Union[Sequence[Sequence[int]], np.ndarray],
    *,
    as_array: bool = False,
) -> tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float],
    float,
    float,
    float,
    float,
    Sequence[Sequence[float]],
    Sequence[Sequence[float]],
]:
    coords_arr = np.asarray(coordinates_inside_cell, dtype=float).reshape(-1, 2)
    contour_arr = np.asarray(contour, dtype=float).reshape(-1, 2)
    center_arr = np.array([center_x, center_y])

    Sigma = np.cov(X)
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    if eigenvalues[1] < eigenvalues[0]:
        Q = np.array([eigenvectors[1], eigenvectors[0]])
        U = (coords_arr @ Q)[:, ::-1]
        contour_U = (contour_arr[:, ::-1] @ Q)[:, ::-1]
        u1_c, u2_c = center_arr @ Q
    else:
        Q = np.array([eigenvectors[0], eigenvectors[1]])
        U = coords_arr[:, ::-1] @ Q
        contour_U = contour_arr @ Q
        u2_c, u1_c = center_arr @ Q

    u1 = U[:, 1]
    u2 = U[:, 0]
    u1_contour = contour_U[:, 1]
    u2_contour = contour_U[:, 0]
    min_u1 = float(u1.min())
    max_u1 = float(u1.max())
    if as_array:
        return (
            u1,
            u2,
            u1_contour,
            u2_contour,
            min_u1,
            max_u1,
            float(u1_c),
            float(u2_c),
            U,
            contour_U,
        )
    return (
        u1.tolist(),
        u2.tolist(),
        u1_contour.tolist(),
        u2_contour.tolist(),
        min_u1,
        max_u1,
        float(u1_c),
        float(u2_c),
        U.tolist(),
        contour_U.tolist(),
    )


def calculate_heatmap_path_vector(
    image_fluo_raw: bytes, contour_raw: bytes, degree: int = 4
) -> list[tuple[float, float]]:
    image_fluo = cv2.imdecode(np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR)
    if image_fluo is None:
        raise ValueError("Failed to decode image")
    image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

    contour = pickle.loads(contour_raw)
    contour_array = np.asarray(contour)
    if contour_array.ndim == 3 and contour_array.shape[1] == 1:
        contour_points = contour_array[:, 0, :]
        contour_for_mask = contour_array.astype(np.int32)
    elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
        contour_points = contour_array
        contour_for_mask = contour_array.reshape(-1, 1, 2).astype(np.int32)
    else:
        raise ValueError("Invalid contour format")

    mask = np.zeros_like(image_fluo_gray)
    cv2.fillPoly(mask, [contour_for_mask], 255)

    y_idx, x_idx = np.nonzero(mask)
    if x_idx.size == 0:
        raise ValueError("No points inside contour")
    points_inside_cell = image_fluo_gray[y_idx, x_idx].astype(float)
    coords_inside_cell = np.column_stack((y_idx, x_idx))
    X = np.vstack((x_idx, y_idx))

    (
        u1,
        u2,
        _u1_contour,
        _u2_contour,
        min_u1,
        max_u1,
        _u1_c,
        _u2_c,
        U,
        _contour_U,
    ) = _basis_conversion(
        contour_points,
        X,
        image_fluo.shape[0] / 2,
        image_fluo.shape[1] / 2,
        coords_inside_cell,
        as_array=True,
    )

    theta = _poly_fit(U, degree=degree)
    if theta.size == 0:
        return []
    poly = np.poly1d(theta)
    poly_der = np.polyder(poly)

    projected_u1 = np.empty_like(points_inside_cell, dtype=float)
    for idx, (u1_val, u2_val) in enumerate(zip(u1, u2)):
        _dist, min_point = _find_minimum_distance_point(
            theta,
            float(u1_val),
            float(u2_val),
            float(min_u1),
            float(max_u1),
            poly=poly,
            poly_der=poly_der,
        )
        projected_u1[idx] = min_point[0]

    if projected_u1.size == 0:
        return []

    sort_idx = np.argsort(projected_u1, kind="mergesort")
    projected_u1 = projected_u1[sort_idx]
    points_inside_cell = points_inside_cell[sort_idx]

    split_num = 35
    delta_l = (max_u1 - min_u1) / split_num if split_num > 0 else 0
    if delta_l == 0:
        return list(zip(projected_u1.tolist(), points_inside_cell.tolist()))

    first_point = (float(projected_u1[0]), float(points_inside_cell[0]))
    last_point = (float(projected_u1[-1]), float(points_inside_cell[-1]))
    path: list[tuple[float, float]] = [first_point]

    bin_idx = np.floor((projected_u1 - min_u1) / delta_l).astype(int)
    bin_idx = np.clip(bin_idx, 0, split_num - 1)
    max_val = np.full(split_num, -np.inf)
    max_idx = np.full(split_num, -1, dtype=int)
    for idx, (b_idx, intensity) in enumerate(zip(bin_idx, points_inside_cell)):
        if b_idx == 0:
            continue
        if intensity > max_val[b_idx]:
            max_val[b_idx] = intensity
            max_idx[b_idx] = idx

    for b_idx in range(1, int(split_num)):
        idx = max_idx[b_idx]
        if idx != -1:
            path.append((float(projected_u1[idx]), float(points_inside_cell[idx])))
    path.append(last_point)

    return path


def build_heatmap_vectors_csv(
    paths: Sequence[Sequence[tuple[float, float]]],
) -> bytes:
    rows: list[list[float | str]] = []
    for path in paths:
        if not path:
            continue
        rows.append([pair[0] for pair in path])
        rows.append([pair[1] for pair in path])

    if not rows:
        return b""

    max_len = max(len(row) for row in rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        padded = row + [""] * (max_len - len(row))
        writer.writerow(padded)
    return buffer.getvalue().encode("utf-8")
