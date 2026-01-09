import io
import pickle
from typing import Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import MetaData, String, Table, cast, or_, select

from app.database_manager.crud import get_database_session, _build_map256_normalized
from app.bulk_engine.heatmap_bulk_core import (
    build_heatmap_vectors_csv,
    calculate_heatmap_path_vector,
)
from app.bulk_engine.hu_separation_detector import build_hu_separation_overlay

PIXEL_SIZE_UM = 0.065


def _pca_length(points: np.ndarray) -> float:
    pts = points.astype(float)
    if pts.shape[0] < 2:
        return 0.0
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ axis
    return float(proj.max() - proj.min())


def _calc_cell_length_um(image_ph: Optional[bytes], contour_raw: bytes) -> float:
    """
    Calculate the major-axis length (um) using PCA.
    Prefer pixels inside the contour; fall back to contour points.
    """
    try:
        contour = pickle.loads(contour_raw)
    except Exception:
        return 0.0

    contour_pts = np.array([p[0] if len(p) == 1 else p for p in contour], dtype=float)
    if contour_pts.ndim != 2 or contour_pts.shape[0] == 0:
        return 0.0

    if image_ph is not None:
        image_ph_gray = cv2.imdecode(np.frombuffer(image_ph, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image_ph_gray is not None:
            mask = np.zeros_like(image_ph_gray)
            cv2.fillPoly(mask, [np.array(contour_pts, dtype=np.int32)], 255)
            coords_inside = np.column_stack(np.where(mask))
            if coords_inside.size > 0:
                length_px = _pca_length(coords_inside[:, ::-1])
                if length_px > 0:
                    return round(length_px * PIXEL_SIZE_UM, 4)

    length_px = _pca_length(contour_pts)
    return round(length_px * PIXEL_SIZE_UM, 4) if length_px > 0 else 0.0


def _get_points_inside_cell(image_raw: bytes, contour_raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return np.array([])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image_gray)
    contour = pickle.loads(contour_raw)
    if isinstance(contour, (list, tuple)):
        contours = contour
    else:
        contours = [contour]
    contours = [np.array(c, dtype=np.int32) for c in contours if c is not None]
    if not contours:
        return np.array([])
    cv2.fillPoly(mask, contours, 255)
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return np.array([])
    return image_gray[coords[:, 0], coords[:, 1]].flatten()


def _calc_normalized_median_intensity(image_raw: bytes, contour_raw: bytes) -> Optional[float]:
    points = _get_points_inside_cell(image_raw, contour_raw)
    if points.size == 0:
        return None
    max_val = float(points.max())
    if max_val <= 0:
        return 0.0
    normalized = points.astype(float) / max_val
    median_val = float(np.median(normalized))
    return round(median_val, 4)


def get_cell_lengths_by_label(
    db_name: str, label: Optional[str] = None
) -> list[tuple[str, float]]:
    """
    Return cell lengths (um) for cells matching the manual_label.
    """
    label_str = str(label).strip() if label is not None else ""
    apply_filter = bool(label_str) and label_str.lower() != "all"

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = (
            select(cells.c.cell_id, cells.c.img_ph, cells.c.contour, cells.c.manual_label)
            .where(cells.c.contour.is_not(None))
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )

        if apply_filter:
            filters = [cast(cells.c.manual_label, String) == label_str]
            if label_str.isdigit():
                filters.append(cells.c.manual_label == int(label_str))
            if label_str.upper() == "N/A":
                filters.append(cells.c.manual_label == "N/A")
                filters.append(cells.c.manual_label == 1000)
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        lengths: list[tuple[str, float]] = []
        for cell_id, image_ph, contour_raw, _ in result.fetchall():
            if cell_id is None or contour_raw is None:
                continue
            image_bytes = bytes(image_ph) if image_ph is not None else None
            contour_bytes = bytes(contour_raw)
            length_val = _calc_cell_length_um(image_bytes, contour_bytes)
            if length_val > 0:
                lengths.append((str(cell_id), length_val))
        return lengths
    finally:
        session.close()


def get_cell_areas_by_label(
    db_name: str, label: Optional[str] = None
) -> list[tuple[str, float]]:
    """
    Return cell areas for cells matching the manual_label.
    """
    label_str = str(label).strip() if label is not None else ""
    apply_filter = bool(label_str) and label_str.lower() != "all"

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = (
            select(cells.c.cell_id, cells.c.area, cells.c.manual_label)
            .where(cells.c.area.is_not(None))
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )

        if apply_filter:
            filters = [cast(cells.c.manual_label, String) == label_str]
            if label_str.isdigit():
                filters.append(cells.c.manual_label == int(label_str))
            if label_str.upper() == "N/A":
                filters.append(cells.c.manual_label == "N/A")
                filters.append(cells.c.manual_label == 1000)
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        areas: list[tuple[str, float]] = []
        for cell_id, area, _ in result.fetchall():
            if cell_id is None or area is None:
                continue
            try:
                area_val = float(area)
            except (TypeError, ValueError):
                continue
            if area_val > 0:
                areas.append((str(cell_id), area_val))
        return areas
    finally:
        session.close()


def get_normalized_medians_by_label(
    db_name: str, label: Optional[str] = None, channel: str = "ph"
) -> list[tuple[str, float]]:
    """
    Return normalized median intensity values for cells matching the manual_label.
    """
    label_str = str(label).strip() if label is not None else ""
    apply_filter = bool(label_str) and label_str.lower() != "all"
    column_map = {
        "ph": "img_ph",
        "fluo1": "img_fluo1",
        "fluo2": "img_fluo2",
    }
    column_name = column_map.get(channel)
    if column_name is None:
        raise ValueError("Invalid channel")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = (
            select(cells.c.cell_id, cells.c[column_name], cells.c.contour, cells.c.manual_label)
            .where(cells.c[column_name].is_not(None))
            .where(cells.c.contour.is_not(None))
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )

        if apply_filter:
            filters = [cast(cells.c.manual_label, String) == label_str]
            if label_str.isdigit():
                filters.append(cells.c.manual_label == int(label_str))
            if label_str.upper() == "N/A":
                filters.append(cells.c.manual_label == "N/A")
                filters.append(cells.c.manual_label == 1000)
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        medians: list[tuple[str, float]] = []
        for cell_id, image_raw, contour_raw, _ in result.fetchall():
            if cell_id is None or image_raw is None or contour_raw is None:
                continue
            median_val = _calc_normalized_median_intensity(bytes(image_raw), bytes(contour_raw))
            if median_val is None or not np.isfinite(median_val):
                continue
            if median_val < 0:
                continue
            medians.append((str(cell_id), float(median_val)))
        return medians
    finally:
        session.close()


def get_raw_intensities_by_label(
    db_name: str, label: Optional[str] = None, channel: str = "ph"
) -> list[tuple[str, list[int]]]:
    """
    Return raw intensity values inside each cell contour for the specified channel.
    """
    label_str = str(label).strip() if label is not None else ""
    apply_filter = bool(label_str) and label_str.lower() != "all"
    column_map = {
        "ph": "img_ph",
        "fluo1": "img_fluo1",
        "fluo2": "img_fluo2",
    }
    column_name = column_map.get(channel)
    if column_name is None:
        raise ValueError("Invalid channel")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = (
            select(cells.c.cell_id, cells.c[column_name], cells.c.contour, cells.c.manual_label)
            .where(cells.c[column_name].is_not(None))
            .where(cells.c.contour.is_not(None))
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )

        if apply_filter:
            filters = [cast(cells.c.manual_label, String) == label_str]
            if label_str.isdigit():
                filters.append(cells.c.manual_label == int(label_str))
            if label_str.upper() == "N/A":
                filters.append(cells.c.manual_label == "N/A")
                filters.append(cells.c.manual_label == 1000)
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        intensities_by_cell: list[tuple[str, list[int]]] = []
        for cell_id, image_raw, contour_raw, _ in result.fetchall():
            if cell_id is None or image_raw is None or contour_raw is None:
                continue
            points = _get_points_inside_cell(bytes(image_raw), bytes(contour_raw))
            values = points.astype(int).tolist() if points.size > 0 else []
            intensities_by_cell.append((str(cell_id), values))
        return intensities_by_cell
    finally:
        session.close()


def _collect_heatmap_paths(
    db_name: str,
    label: Optional[str] = None,
    channel: str = "fluo1",
    degree: int = 4,
) -> list[list[tuple[float, float]]]:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    label_str = str(label).strip() if label is not None else ""
    apply_filter = bool(label_str) and label_str.lower() != "all"
    column_map = {
        "fluo1": "img_fluo1",
        "fluo2": "img_fluo2",
    }
    column_name = column_map.get(channel)
    if column_name is None:
        raise ValueError("Invalid channel")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = (
            select(cells.c.cell_id, cells.c[column_name], cells.c.contour, cells.c.manual_label)
            .where(cells.c[column_name].is_not(None))
            .where(cells.c.contour.is_not(None))
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )

        if apply_filter:
            filters = [cast(cells.c.manual_label, String) == label_str]
            if label_str.isdigit():
                filters.append(cells.c.manual_label == int(label_str))
            if label_str.upper() == "N/A":
                filters.append(cells.c.manual_label == "N/A")
                filters.append(cells.c.manual_label == 1000)
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        paths: list[list[tuple[float, float]]] = []
        for _cell_id, image_raw, contour_raw, _ in result.fetchall():
            if image_raw is None or contour_raw is None:
                continue
            try:
                path = calculate_heatmap_path_vector(
                    bytes(image_raw),
                    bytes(contour_raw),
                    degree=degree,
                )
            except Exception:
                continue
            if path:
                paths.append(path)

        if not paths:
            raise LookupError("No heatmap vectors found for the specified label.")
        return paths
    finally:
        session.close()


def _build_heatmap_abs_plot(
    paths: Sequence[Sequence[tuple[float, float]]], dpi: int = 100
) -> bytes:
    heatmap_vectors: list[dict[str, object]] = []
    for idx, path in enumerate(paths):
        if not path:
            continue
        u1_values = [float(pair[0]) for pair in path]
        g_values = [float(pair[1]) for pair in path]
        if not u1_values or not g_values:
            continue
        count = min(len(u1_values), len(g_values))
        u1_values = u1_values[:count]
        g_values = g_values[:count]
        min_u1 = min(u1_values)
        max_u1 = max(u1_values)
        length = max_u1 - min_u1
        heatmap_vectors.append(
            {
                "index": idx,
                "u1": [val - min_u1 for val in u1_values],
                "G": g_values,
                "length": length,
            }
        )

    if not heatmap_vectors:
        raise LookupError("No heatmap vectors found for the specified label.")

    heatmap_vectors.sort(key=lambda vec: float(vec["length"]))
    max_length = max(float(vec["length"]) for vec in heatmap_vectors)

    for vec in heatmap_vectors:
        length = float(vec["length"])
        offset = (max_length - length) / 2 - max_length / 2
        vec["u1"] = [float(val) + offset for val in vec["u1"]]

    u1_all = [val for vec in heatmap_vectors for val in vec["u1"]]
    if not u1_all:
        raise LookupError("No heatmap vectors found for the specified label.")
    u1_min = min(u1_all)
    u1_max = max(u1_all)

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = plt.cm.jet

    for idx, vec in enumerate(heatmap_vectors):
        u1 = vec["u1"]
        g_values = vec["G"]
        count = min(len(u1), len(g_values))
        if count < 2:
            continue
        u1 = u1[:count]
        g_values = g_values[:count]
        g_array = np.array(g_values, dtype=float)
        g_min = float(np.min(g_array))
        g_max = float(np.max(g_array))
        if g_max == g_min:
            normalized = np.zeros_like(g_array)
        else:
            normalized = (g_array - g_min) / (g_max - g_min)
        colors = cmap(normalized)

        offset = len(heatmap_vectors) - idx - 1
        for i in range(len(u1) - 1):
            ax.plot([offset, offset], u1[i : i + 2], color=colors[i], lw=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Normalized G Value")

    ax.set_ylim([u1_min, u1_max])
    ax.set_xlim([-0.5, len(heatmap_vectors) - 0.5])
    ax.set_ylabel("Cell length (px)")
    ax.set_xlabel("Cell number")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _build_heatmap_rel_plot(
    paths: Sequence[Sequence[tuple[float, float]]], dpi: int = 100
) -> bytes:
    heatmap_vectors: list[dict[str, object]] = []
    for idx, path in enumerate(paths):
        if not path:
            continue
        g_values = [float(pair[1]) for pair in path]
        if not g_values:
            continue
        length = len(g_values)
        heatmap_vectors.append(
            {
                "index": idx,
                "u1": list(range(length)),
                "G": g_values,
                "length": length,
                "g_sum": float(np.sum(g_values)),
            }
        )

    if not heatmap_vectors:
        raise LookupError("No heatmap vectors found for the specified label.")

    heatmap_vectors.sort(key=lambda vec: float(vec["g_sum"]))
    max_length = max(int(vec["length"]) for vec in heatmap_vectors)

    for vec in heatmap_vectors:
        length = int(vec["length"])
        offset = (max_length - length) / 2 - max_length / 2
        vec["u1"] = [float(val) + offset for val in vec["u1"]]

    u1_all = [val for vec in heatmap_vectors for val in vec["u1"]]
    if not u1_all:
        raise LookupError("No heatmap vectors found for the specified label.")
    u1_min = min(u1_all)
    u1_max = max(u1_all)

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = plt.cm.jet

    for idx, vec in enumerate(heatmap_vectors):
        u1 = vec["u1"]
        g_values = vec["G"]
        count = min(len(u1), len(g_values))
        if count < 2:
            continue
        u1 = u1[:count]
        g_values = g_values[:count]
        g_array = np.array(g_values, dtype=float)
        g_min = float(np.min(g_array))
        g_max = float(np.max(g_array))
        if g_max == g_min:
            normalized = np.zeros_like(g_array)
        else:
            normalized = (g_array - g_min) / (g_max - g_min)
        colors = cmap(normalized)

        offset = len(heatmap_vectors) - idx - 1
        for i in range(len(u1) - 1):
            ax.plot([offset, offset], u1[i : i + 2], color=colors[i], lw=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Normalized G Value")

    ax.set_ylim([u1_min, u1_max])
    ax.set_xlim([-0.5, len(heatmap_vectors) - 0.5])
    ax.set_ylabel("Relative position(-)")
    ax.set_xlabel("Cell number")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_heatmap_vectors_csv(
    db_name: str,
    label: Optional[str] = None,
    channel: str = "fluo1",
    degree: int = 4,
) -> bytes:
    """
    Return CSV bytes for heatmap vectors (u1/G pairs per cell).
    """
    paths = _collect_heatmap_paths(db_name, label=label, channel=channel, degree=degree)
    csv_bytes = build_heatmap_vectors_csv(paths)
    if not csv_bytes:
        raise LookupError("No heatmap vectors found for the specified label.")
    return csv_bytes


def create_heatmap_abs_plot(
    db_name: str,
    label: Optional[str] = None,
    channel: str = "fluo1",
    degree: int = 4,
) -> bytes:
    paths = _collect_heatmap_paths(db_name, label=label, channel=channel, degree=degree)
    return _build_heatmap_abs_plot(paths)


def create_heatmap_rel_plot(
    db_name: str,
    label: Optional[str] = None,
    channel: str = "fluo1",
    degree: int = 4,
) -> bytes:
    paths = _collect_heatmap_paths(db_name, label=label, channel=channel, degree=degree)
    return _build_heatmap_rel_plot(paths)


def create_hu_separation_overlay(
    db_name: str,
    label: Optional[str] = None,
    channel: str = "fluo1",
    degree: int = 4,
    center_ratio: float = 0.15,
    max_to_min_ratio: float = 0.9,
) -> bytes:
    paths = _collect_heatmap_paths(db_name, label=label, channel=channel, degree=degree)
    csv_bytes = build_heatmap_vectors_csv(paths)
    if not csv_bytes:
        raise LookupError("No heatmap vectors found for the specified label.")
    filename = f"heatmap-{db_name}"
    buf = build_hu_separation_overlay(
        [(filename, csv_bytes)],
        degree=degree,
        center_ratio=center_ratio,
        max_to_min_ratio=max_to_min_ratio,
    )
    return buf.getvalue()


def create_map256_strip(
    db_name: str,
    label: Optional[str] = None,
    channel: str = "fluo1",
    degree: int = 4,
) -> bytes:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    label_str = str(label).strip() if label is not None else ""
    apply_filter = bool(label_str) and label_str.lower() != "all"
    column_map = {
        "fluo1": "img_fluo1",
        "fluo2": "img_fluo2",
    }
    column_name = column_map.get(channel)
    if column_name is None:
        raise ValueError("Invalid channel")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = (
            select(cells.c.cell_id, cells.c[column_name], cells.c.contour, cells.c.manual_label)
            .where(cells.c[column_name].is_not(None))
            .where(cells.c.contour.is_not(None))
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )

        if apply_filter:
            filters = [cast(cells.c.manual_label, String) == label_str]
            if label_str.isdigit():
                filters.append(cells.c.manual_label == int(label_str))
            if label_str.upper() == "N/A":
                filters.append(cells.c.manual_label == "N/A")
                filters.append(cells.c.manual_label == 1000)
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        rotated_images: list[np.ndarray] = []
        for _cell_id, image_raw, contour_raw, _ in result.fetchall():
            if image_raw is None or contour_raw is None:
                continue
            try:
                normalized = _build_map256_normalized(bytes(image_raw), bytes(contour_raw), degree)
            except Exception:
                continue
            rotated = cv2.rotate(normalized, cv2.ROTATE_90_CLOCKWISE)
            rotated_images.append(rotated)

        if not rotated_images:
            raise LookupError("No map256 images found for the specified label.")

        combined = (
            cv2.hconcat(rotated_images)
            if len(rotated_images) > 1
            else rotated_images[0]
        )
        success, buffer = cv2.imencode(".png", combined)
        if not success:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()
    finally:
        session.close()


def create_cell_length_boxplot(
    db_name: str, label: Optional[str] = None
) -> bytes:
    lengths = get_cell_lengths_by_label(db_name, label)
    if not lengths:
        raise LookupError("No cells found for the specified label.")
    values = [length for _, length in lengths]

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=180)
    ax.boxplot(
        values,
        vert=True,
        widths=0.35,
        patch_artist=False,
        showfliers=False,
        boxprops={"color": "#4a5568", "linewidth": 1.2},
        medianprops={"color": "#2f855a", "linewidth": 1.4},
        whiskerprops={"color": "#4a5568", "linewidth": 1.1},
        capprops={"color": "#4a5568", "linewidth": 1.1},
    )
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(values))
    ax.scatter(
        1 + jitter,
        values,
        s=16,
        color="#2c7a7b",
        alpha=0.65,
        linewidth=0,
    )
    label_text = str(label) if label not in (None, "", "all", "All") else "All"
    ax.set_xticks([1])
    ax.set_xticklabels([label_text])
    ax.set_ylabel("Cell length (um)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_title(f"{db_name} | label {label_text}", fontsize=9)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_cell_area_boxplot(
    db_name: str, label: Optional[str] = None
) -> bytes:
    areas = get_cell_areas_by_label(db_name, label)
    if not areas:
        raise LookupError("No cells found for the specified label.")
    values = [area for _, area in areas]

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=180)
    ax.boxplot(
        values,
        vert=True,
        widths=0.35,
        patch_artist=False,
        showfliers=False,
        boxprops={"color": "#4a5568", "linewidth": 1.2},
        medianprops={"color": "#b83280", "linewidth": 1.4},
        whiskerprops={"color": "#4a5568", "linewidth": 1.1},
        capprops={"color": "#4a5568", "linewidth": 1.1},
    )
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(values))
    ax.scatter(
        1 + jitter,
        values,
        s=16,
        color="#b7791f",
        alpha=0.65,
        linewidth=0,
    )
    label_text = str(label) if label not in (None, "", "all", "All") else "All"
    ax.set_xticks([1])
    ax.set_xticklabels([label_text])
    ax.set_ylabel("Cell area (px^2)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_title(f"{db_name} | label {label_text}", fontsize=9)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_normalized_median_boxplot(
    db_name: str, label: Optional[str] = None, channel: str = "ph"
) -> bytes:
    medians = get_normalized_medians_by_label(db_name, label, channel)
    if not medians:
        raise LookupError("No cells found for the specified label.")
    values = [median for _, median in medians]

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=180)
    ax.boxplot(
        values,
        vert=True,
        widths=0.35,
        patch_artist=False,
        showfliers=False,
        boxprops={"color": "#2d3748", "linewidth": 1.2},
        medianprops={"color": "#2b6cb0", "linewidth": 1.4},
        whiskerprops={"color": "#2d3748", "linewidth": 1.1},
        capprops={"color": "#2d3748", "linewidth": 1.1},
    )
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(values))
    ax.scatter(
        1 + jitter,
        values,
        s=16,
        color="#2b6cb0",
        alpha=0.65,
        linewidth=0,
    )
    label_text = str(label) if label not in (None, "", "all", "All") else "All"
    ax.set_xticks([1])
    ax.set_xticklabels([label_text])
    ax.set_ylabel("Normalized median intensity")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_title(f"{db_name} | label {label_text} | {channel}", fontsize=9)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
