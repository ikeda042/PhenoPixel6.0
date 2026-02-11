import io
import json
from pathlib import Path
import pickle
from typing import AsyncIterator, Literal, Sequence
import zipfile

import aiofiles
from aiofiles import os as aioos
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import BLOB, FLOAT, Column, Integer, MetaData, String, Table, create_engine, or_, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeMeta, Session, declarative_base, sessionmaker

DATABASES_DIR: Path = Path(__file__).resolve().parents[1] / "databases"
DOWNLOAD_CHUNK_SIZE: int = 1024 * 1024
ANNOTATION_DOWNSCALE: float = 0.2
ANNOTATION_CONTOUR_THICKNESS: int = 3
FLUO_COLOR_CHANNELS: dict[str, tuple[int, int, int]] = {
    "blue": (1, 0, 0),
    "green": (0, 1, 0),
    "red": (0, 0, 1),
    "yellow": (0, 1, 1),
    "magenta": (1, 0, 1),
    "gray": (1, 1, 1),
}
FLUO_DEFAULT_BY_TYPE: dict[str, str] = {"fluo1": "green", "fluo2": "magenta"}


def _resolve_fluo_color(
    fluo_color: str | None, image_type: Literal["fluo1", "fluo2"]
) -> str:
    color_key = (fluo_color or "").strip().lower()
    if not color_key:
        color_key = FLUO_DEFAULT_BY_TYPE.get(image_type, "green")
    if color_key not in FLUO_COLOR_CHANNELS:
        raise ValueError("Invalid fluo_color")
    return color_key


def _apply_fluo_overlay(
    overlay: np.ndarray, intensity: np.ndarray, mask: np.ndarray, color_key: str
) -> None:
    blue, green, red = FLUO_COLOR_CHANNELS[color_key]
    if blue:
        channel = overlay[:, :, 0]
        channel[mask] = np.maximum(channel[mask], intensity[mask])
        overlay[:, :, 0] = channel
    if green:
        channel = overlay[:, :, 1]
        channel[mask] = np.maximum(channel[mask], intensity[mask])
        overlay[:, :, 1] = channel
    if red:
        channel = overlay[:, :, 2]
        channel[mask] = np.maximum(channel[mask], intensity[mask])
        overlay[:, :, 2] = channel

Base: DeclarativeMeta = declarative_base()


class Cell(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB, nullable=True)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    user_id = Column(String, nullable=True)


def _sanitize_db_name(db_name: str) -> str:
    cleaned = Path(db_name or "").name.strip()
    if not cleaned:
        raise ValueError("Database name is required")
    if not cleaned.endswith(".db"):
        raise ValueError("Database name must end with .db")
    return cleaned


def sanitize_db_name(db_name: str) -> str:
    return _sanitize_db_name(db_name)


def resolve_database_path(db_name: str) -> Path:
    sanitized = _sanitize_db_name(db_name)
    return DATABASES_DIR / sanitized


async def read_database_chunks(
    path: Path, chunk_size: int = DOWNLOAD_CHUNK_SIZE
) -> AsyncIterator[bytes]:
    async with aiofiles.open(path, "rb") as file_handle:
        while True:
            chunk = await file_handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def delete_database(db_name: str) -> str:
    db_path = resolve_database_path(db_name)
    if not db_path.is_file():
        raise FileNotFoundError("Database not found")
    await aioos.remove(db_path)
    return db_path.name


async def rename_database(old_name: str, new_name: str) -> tuple[str, str]:
    old_cleaned = _sanitize_db_name(old_name)
    new_cleaned = _sanitize_db_name(new_name)
    if old_cleaned == new_cleaned:
        return old_cleaned, new_cleaned
    old_path = DATABASES_DIR / old_cleaned
    if not old_path.is_file():
        raise FileNotFoundError("Database not found")
    new_path = DATABASES_DIR / new_cleaned
    if new_path.exists():
        raise FileExistsError("Database name already exists")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar_new = DATABASES_DIR / f"{new_cleaned}{suffix}"
        if sidecar_new.exists():
            raise FileExistsError("Database name already exists")
    await aioos.rename(old_path, new_path)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar_old = DATABASES_DIR / f"{old_cleaned}{suffix}"
        if sidecar_old.is_file():
            await aioos.rename(sidecar_old, DATABASES_DIR / f"{new_cleaned}{suffix}")
    return old_cleaned, new_cleaned


def get_database_session(db_name: str) -> Session:
    """
    Return a synchronous SQLAlchemy session for the given database file.
    The caller is responsible for closing the session.
    """
    cleaned = _sanitize_db_name(db_name)
    db_path = DATABASES_DIR / cleaned
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {cleaned}")
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    return sessionmaker(engine, expire_on_commit=False)()


def migrate_database(db_name: str) -> None:
    cleaned = _sanitize_db_name(db_name)
    db_path = DATABASES_DIR / cleaned
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {cleaned}")
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        with engine.connect() as connection:
            inspector = engine.dialect.get_columns(connection, Cell.__tablename__)
            existing_columns = {col["name"] for col in inspector}

            if "img_fluo" in existing_columns:
                try:
                    connection.execute(
                        text(
                            f"ALTER TABLE {Cell.__tablename__} RENAME COLUMN img_fluo TO img_fluo1"
                        )
                    )
                    existing_columns.discard("img_fluo")
                    existing_columns.add("img_fluo1")
                except OperationalError as exc:
                    print(f"Failed to rename column 'img_fluo': {exc}")

            model_columns = {col.name for col in Cell.__table__.columns}
            missing_columns = model_columns - existing_columns

            if missing_columns:
                for col_name in missing_columns:
                    col_type = next(
                        (
                            col.type
                            for col in Cell.__table__.columns
                            if col.name == col_name
                        ),
                        None,
                    )
                    if col_type is None:
                        continue
                    alter_query = (
                        f"ALTER TABLE {Cell.__tablename__} "
                        f"ADD COLUMN {col_name} {col_type}"
                    )
                    try:
                        connection.execute(text(alter_query))
                    except OperationalError as exc:
                        print(f"Failed to add column '{col_name}': {exc}")
                session.commit()
    finally:
        session.close()


async def list_databases() -> list[str]:
    if not DATABASES_DIR.is_dir():
        return []
    entries = await aioos.listdir(DATABASES_DIR)
    return sorted(
        [
            name
            for name in entries
            if (DATABASES_DIR / name).is_file() and name.endswith(".db")
        ]
    )


def get_cell_ids(db_name: str) -> list[str]:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(cells.c.cell_id)
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )
        result = session.execute(stmt)
        return [row[0] for row in result.fetchall()]
    finally:
        session.close()


def get_cell_label(db_name: str, cell_id: str) -> str:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(cells.c.manual_label)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        label = session.execute(stmt).scalar_one_or_none()
        if label is None:
            raise LookupError("Cell label not found")
        return str(label)
    finally:
        session.close()


def update_cell_label(db_name: str, cell_id: str, label: str) -> str:
    label_value = str(label or "").strip()
    if not label_value:
        raise ValueError("Label is required")
    if label_value.isdigit():
        normalized_label = int(label_value)
    else:
        normalized_label = label_value

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            update(cells)
            .where(cells.c.cell_id == cell_id)
            .values(manual_label=normalized_label)
        )
        result = session.execute(stmt)
        if result.rowcount == 0:
            raise LookupError("Cell label not found")
        session.commit()
    finally:
        session.close()
    return str(normalized_label)


def get_cell_contour(db_name: str, cell_id: str) -> list[list[float]]:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        contour_raw = session.execute(stmt).scalar_one_or_none()
        if contour_raw is None:
            raise LookupError("Cell contour not found")
        contour = pickle.loads(contour_raw)
        contour_array = np.asarray(contour)
        if contour_array.ndim == 3 and contour_array.shape[1] == 1:
            contour_array = contour_array[:, 0, :]
        if contour_array.ndim != 2 or contour_array.shape[1] != 2:
            raise ValueError("Invalid contour format")
        return contour_array.astype(float).tolist()
    finally:
        session.close()


def apply_elastic_contour(
    db_name: str, cell_id: str, delta: int = 0
) -> list[list[float]]:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(cells.c.img_ph, cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None:
            raise LookupError("Cell contour not found")
        image_blob, contour_blob = row
        if image_blob is None or contour_blob is None:
            raise LookupError("Cell contour not found")

        image = _decode_image(bytes(image_blob))
        contour_raw = bytes(contour_blob)
        contour = pickle.loads(contour_raw)
        contour_array = np.asarray(contour)
        if contour_array.ndim == 3 and contour_array.shape[1] == 1:
            contour_np = contour_array.astype(np.int32)
        elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
            contour_np = contour_array.reshape(-1, 1, 2).astype(np.int32)
        else:
            raise ValueError("Invalid contour format")

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour_np], -1, 255, -1)
        kernel = np.ones((3, 3), np.uint8)
        if delta > 0:
            mask = cv2.dilate(mask, kernel, iterations=delta)
        elif delta < 0:
            mask = cv2.erode(mask, kernel, iterations=-delta)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            new_contour = contour_np
        else:
            moments = cv2.moments(contour_np)
            cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else 0
            cy = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else 0
            min_distance = float("inf")
            new_contour = contours[0]
            for candidate in contours:
                m2 = cv2.moments(candidate)
                if m2["m00"] == 0:
                    continue
                cx2 = int(m2["m10"] / m2["m00"])
                cy2 = int(m2["m01"] / m2["m00"])
                distance = float(np.hypot(cx - cx2, cy - cy2))
                if distance < min_distance:
                    min_distance = distance
                    new_contour = candidate

        contour_to_store = new_contour.reshape(-1, 1, 2).astype(np.int32)
        pickled_contour = pickle.dumps(contour_to_store)
        update_stmt = (
            update(cells)
            .where(cells.c.cell_id == cell_id)
            .values(contour=pickled_contour)
        )
        result = session.execute(update_stmt)
        if result.rowcount == 0:
            raise LookupError("Cell contour not found")
        session.commit()

        flat = contour_to_store.reshape(-1, 2).astype(float)
        return flat.tolist()
    finally:
        session.close()


def apply_elastic_contour_bulk(
    db_name: str, delta: int = 0, label: str | None = None
) -> dict:
    if label is None:
        cell_ids = get_cell_ids(db_name)
    else:
        cell_ids = get_cell_ids_by_label(db_name, label)
    if not cell_ids:
        raise LookupError("No cells found")

    updated = 0
    failed = 0
    failures: list[dict[str, str]] = []
    for cell_id in cell_ids:
        try:
            apply_elastic_contour(db_name, cell_id, delta)
            updated += 1
        except Exception as exc:
            failed += 1
            if len(failures) < 25:
                failures.append({"cell_id": cell_id, "error": str(exc)})

    return {
        "total": len(cell_ids),
        "updated": updated,
        "failed": failed,
        "failures": failures,
    }


def get_cell_ids_by_label(db_name: str, label: str) -> list[str]:
    label_value = label.strip()
    if not label_value:
        raise ValueError("Label is required")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        filters = [cells.c.manual_label == label_value]
        if label_value.isdigit():
            filters.append(cells.c.manual_label == int(label_value))

        stmt = (
            select(cells.c.cell_id)
            .where(cells.c.cell_id.is_not(None))
            .where(or_(*filters))
            .order_by(cells.c.cell_id)
        )
        result = session.execute(stmt)
        return [row[0] for row in result.fetchall()]
    finally:
        session.close()


def get_manual_labels(db_name: str) -> list[str]:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)

        stmt = select(cells.c.manual_label).where(cells.c.manual_label.is_not(None)).distinct()
        result = session.execute(stmt).scalars().all()
        labels: list[str] = []
        for value in result:
            if value is None:
                continue
            label = str(value).strip()
            if not label:
                continue
            labels.append(label)
        return sorted(set(labels), key=str.lower)
    finally:
        session.close()


def _decode_image(image_data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _encode_image(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


def _encode_image_with_format(
    image: np.ndarray,
    image_format: str,
    jpeg_quality: int = 80,
) -> bytes:
    fmt = (image_format or "png").lower()
    if fmt in ("jpg", "jpeg"):
        quality = max(10, min(int(jpeg_quality), 95))
        success, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
    else:
        success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


def _colorize_fluo_image(
    image: np.ndarray,
    image_type: Literal["fluo1", "fluo2"],
    fluo_color: str | None = None,
) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_key = _resolve_fluo_color(fluo_color, image_type)
    blue, green, red = FLUO_COLOR_CHANNELS[color_key]
    color = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=gray.dtype)
    if blue:
        color[:, :, 0] = gray
    if green:
        color[:, :, 1] = gray
    if red:
        color[:, :, 2] = gray
    return color


def _normalize_grayscale_to_uint8(image: np.ndarray) -> np.ndarray:
    min_val = float(image.min())
    max_val = float(image.max())
    range_val = max_val - min_val
    if range_val <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image.astype(np.float32) - min_val) / range_val * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _subtract_background(gray_img: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(gray_img, background)


def _flip_image_if_needed(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    height, width = image_gray.shape
    if width < 2:
        return image
    left_half = image_gray[:, : width // 2]
    right_half = image_gray[:, width // 2 :]
    if float(np.mean(right_half)) > float(np.mean(left_half)):
        return cv2.flip(image, 1)
    return image


def _build_arc_length_lookup(
    coefficients: np.ndarray,
    min_x: float,
    max_x: float,
    steps: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(min_x) or not np.isfinite(max_x) or max_x <= min_x:
        return np.array([min_x, max_x], dtype=float), np.array([0.0, 0.0], dtype=float)
    poly = np.poly1d(coefficients)
    poly_der = np.polyder(poly)
    xs = np.linspace(min_x, max_x, steps)
    slopes = poly_der(xs)
    integrand = np.sqrt(1.0 + slopes * slopes)
    cumulative = np.zeros_like(xs)
    if xs.size > 1:
        dx = np.diff(xs)
        cumulative[1:] = np.cumsum((integrand[1:] + integrand[:-1]) * 0.5 * dx)
    return xs, cumulative


def _find_minimum_distance_point(
    coefficients: np.ndarray,
    x_q: float,
    y_q: float,
    min_x: float,
    max_x: float,
) -> tuple[float, tuple[float, float]]:
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


def _draw_contour(image: np.ndarray, contour_raw: bytes, thickness: int = 1) -> np.ndarray:
    contour = pickle.loads(contour_raw)
    return cv2.drawContours(image, contour, -1, (0, 255, 0), thickness)


def _draw_scale_bar_with_centered_text(image: np.ndarray) -> np.ndarray:
    pixels_per_um = 1 / 0.065
    scale_bar_um = 5
    scale_bar_length_px = int(scale_bar_um * pixels_per_um)
    scale_bar_thickness = 2
    scale_bar_color = (255, 255, 255)

    margin = 20
    x1 = image.shape[1] - margin - scale_bar_length_px
    y1 = image.shape[0] - margin
    x2 = x1 + scale_bar_length_px
    y2 = y1 + scale_bar_thickness

    cv2.rectangle(
        image, (x1, y1), (x2, y2), scale_bar_color, thickness=cv2.FILLED
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{scale_bar_um} um"
    text_scale = 0.4
    text_thickness = 0
    text_color = (255, 255, 255)

    text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
    text_x = x1 + (scale_bar_length_px - text_size[0]) // 2
    text_y = y2 + text_size[1] + 5

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        text_scale,
        text_color,
        text_thickness,
    )
    return image


def _pad_to_square(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
    height, width = image.shape[:2]
    if height == width:
        return image
    size = max(height, width)
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top
    pad_left = (size - width) // 2
    pad_right = size - width - pad_left
    if image.ndim == 2:
        return cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=fill_value,
        )
    fill_color = (fill_value, fill_value, fill_value)
    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=fill_color,
    )


def _normalize_annotation_label(label: object) -> str:
    if label is None:
        return "N/A"
    if isinstance(label, (bytes, bytearray)):
        label_str = label.decode(errors="ignore").strip()
    else:
        label_str = str(label).strip()
    if not label_str:
        return "N/A"
    if label_str.upper() == "N/A" or label_str == "1000":
        return "N/A"
    return label_str


def _normalize_fast_manual_label(label: object) -> str | None:
    if label is None:
        return None
    if isinstance(label, (bytes, bytearray)):
        label_str = label.decode(errors="ignore").strip()
    else:
        label_str = str(label).strip()
    return label_str or None


def _safe_cell_filename(cell_id: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in cell_id)
    return safe or "cell"


def _localization_index_energy_1d(intensities: Sequence[float]) -> float:
    vals = [max(float(v), 0.0) for v in intensities]
    count = len(vals)
    if count <= 1:
        return 0.0
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    sq_sum = sum(v * v for v in vals)
    score = 1.0 - (total * total) / (count * sq_sum)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


def _poly_fit(values: Sequence[Sequence[float]], degree: int = 1) -> np.ndarray:
    u1_values = np.array([val[1] for val in values], dtype=float)
    f_values = np.array([val[0] for val in values], dtype=float)
    W = np.vander(u1_values, degree + 1)
    try:
        coefficients = np.linalg.inv(W.T @ W) @ W.T @ f_values
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(W) @ f_values
    return coefficients


def _basis_conversion(
    contour: list[list[int]],
    X: np.ndarray,
    center_x: float,
    center_y: float,
    coordinates_inside_cell: list[list[int]],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    float,
    float,
    float,
    float,
    list[list[float]],
    list[list[float]],
]:
    coords_arr = np.asarray(coordinates_inside_cell).reshape(-1, 2)
    contour_arr = np.asarray(contour).reshape(-1, 2)
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


def _prepare_replot_geometry(
    image_fluo_gray: np.ndarray,
    contour_raw: bytes,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[float]],
]:
    mask = np.zeros_like(image_fluo_gray)
    unpickled_contour = pickle.loads(contour_raw)
    contour_array = np.asarray(unpickled_contour)
    if contour_array.ndim == 3 and contour_array.shape[1] == 1:
        contour_points = contour_array[:, 0, :].tolist()
        contour_for_mask = contour_array.astype(np.int32)
    elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
        contour_points = contour_array.tolist()
        contour_for_mask = contour_array.reshape(-1, 1, 2).astype(np.int32)
    else:
        raise ValueError("Invalid contour format")
    cv2.fillPoly(mask, [contour_for_mask], 255)

    coords_inside_cell = np.column_stack(np.where(mask))
    if coords_inside_cell.size == 0:
        raise ValueError("No points inside contour")

    X = np.array(
        [
            [i[1] for i in coords_inside_cell],
            [i[0] for i in coords_inside_cell],
        ]
    )

    (
        u1,
        u2,
        u1_contour,
        u2_contour,
        _min_u1,
        _max_u1,
        u1_c,
        u2_c,
        U,
        _contour_U,
    ) = _basis_conversion(
        contour_points,
        X,
        image_fluo_gray.shape[0] / 2,
        image_fluo_gray.shape[1] / 2,
        coords_inside_cell.tolist(),
    )

    u1_shifted = np.array(u1) - u1_c
    u2_shifted = np.array(u2) - u2_c
    u1_contour_shifted = np.array(u1_contour) - u1_c
    u2_contour_shifted = np.array(u2_contour) - u2_c

    U_shifted = []
    for y_val, x_val in U:
        x_val_shifted = x_val - u1_c
        y_val_shifted = y_val - u2_c
        U_shifted.append([y_val_shifted, x_val_shifted])

    return (
        coords_inside_cell,
        u1_shifted,
        u2_shifted,
        u1_contour_shifted,
        u2_contour_shifted,
        U_shifted,
    )


def _compute_replot_stats(points_inside_cell: np.ndarray) -> dict[str, float]:
    max_val = np.max(points_inside_cell) if len(points_inside_cell) else 1
    normalized_points = [i / max_val for i in points_inside_cell]
    median_val = float(np.median(points_inside_cell))
    mean_val = float(np.mean(points_inside_cell))
    normalized_median_val = float(np.median(normalized_points))
    normalized_mean_val = float(np.mean(normalized_points))
    sd_val = float(np.std(points_inside_cell))
    cv_val = sd_val / mean_val if mean_val != 0 else 0.0
    localization_val = _localization_index_energy_1d(points_inside_cell)
    return {
        "sd": sd_val,
        "cv": cv_val,
        "localization": localization_val,
        "median": median_val,
        "mean": mean_val,
        "normalized_median": normalized_median_val,
        "normalized_mean": normalized_mean_val,
    }


def _generate_replot_image(
    image_fluo_raw: bytes,
    contour_raw: bytes,
    degree: int,
    dark_mode: bool = False,
) -> bytes:
    image_fluo = cv2.imdecode(np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR)
    if image_fluo is None:
        raise ValueError("Failed to decode image")
    image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)
    (
        coords_inside_cell,
        u1_shifted,
        u2_shifted,
        u1_contour_shifted,
        u2_contour_shifted,
        U_shifted,
    ) = _prepare_replot_geometry(image_fluo_gray, contour_raw)

    points_inside_cell = image_fluo_gray[
        coords_inside_cell[:, 0], coords_inside_cell[:, 1]
    ]

    style = "dark_background" if dark_mode else "default"
    text_kwargs = {"color": "white"} if dark_mode else {}
    margin_width = 20
    margin_height = 20

    with plt.style.context(style):
        fig = plt.figure(figsize=(6, 6))

        plt.scatter(u1_shifted, u2_shifted, s=5, label="Points in cell")
        plt.scatter([0], [0], color="red", s=100, label="Centroid (0,0)")
        plt.axis("equal")
        plt.scatter(
            [val[1] for val in U_shifted],
            [val[0] for val in U_shifted],
            c=points_inside_cell,
            cmap="jet",
            marker="o",
            s=20,
            label="Intensity",
        )

        min_u1_shifted = float(np.min(u1_shifted))
        max_u1_shifted = float(np.max(u1_shifted))
        min_u2_shifted = float(np.min(u2_shifted))
        max_u2_shifted = float(np.max(u2_shifted))
        x_min = min_u1_shifted - margin_width
        x_max = max_u1_shifted + margin_width
        y_min = min_u2_shifted - margin_height
        y_max = max_u2_shifted + margin_height
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        stats = _compute_replot_stats(points_inside_cell)

        plt.text(
            0.5,
            0.35,
            f"SD: {stats['sd']:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )
        plt.text(
            0.5,
            0.30,
            f"CV: {stats['cv']:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )
        plt.text(
            0.5,
            0.25,
            f"Localization score: {stats['localization']:.4g}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )
        plt.text(
            0.5,
            0.20,
            f"Median: {stats['median']:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )
        plt.text(
            0.5,
            0.15,
            f"Mean: {stats['mean']:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )
        plt.text(
            0.5,
            0.10,
            f"Normalized median: {stats['normalized_median']:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )
        plt.text(
            0.5,
            0.05,
            f"Normalized mean: {stats['normalized_mean']:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            **text_kwargs,
        )

        x_for_fit = np.linspace(min_u1_shifted, max_u1_shifted, 1000)
        theta = _poly_fit(U_shifted, degree=degree)
        y_for_fit = np.polyval(theta, x_for_fit)
        plt.plot(x_for_fit, y_for_fit, color="red", label="Poly fit")

        plt.scatter(
            u1_contour_shifted,
            u2_contour_shifted,
            color="lime",
            s=20,
            label="Contour",
        )

        plt.tick_params(direction="in")
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

    return buf.getvalue()


def _generate_replot_overlay_image(
    image_fluo1_raw: bytes,
    image_fluo2_raw: bytes,
    contour_raw: bytes,
    degree: int,
    dark_mode: bool = False,
) -> bytes:
    image_fluo1 = cv2.imdecode(np.frombuffer(image_fluo1_raw, np.uint8), cv2.IMREAD_COLOR)
    if image_fluo1 is None:
        raise ValueError("Failed to decode fluo1 image")
    image_fluo2 = cv2.imdecode(np.frombuffer(image_fluo2_raw, np.uint8), cv2.IMREAD_COLOR)
    if image_fluo2 is None:
        raise ValueError("Failed to decode fluo2 image")
    if image_fluo1.shape[:2] != image_fluo2.shape[:2]:
        raise ValueError("Fluo image dimensions do not match")

    fluo1_gray = cv2.cvtColor(image_fluo1, cv2.COLOR_BGR2GRAY)
    fluo2_gray = cv2.cvtColor(image_fluo2, cv2.COLOR_BGR2GRAY)

    (
        coords_inside_cell,
        u1_shifted,
        u2_shifted,
        u1_contour_shifted,
        u2_contour_shifted,
        U_shifted,
    ) = _prepare_replot_geometry(fluo1_gray, contour_raw)

    points_inside_cell_1 = fluo1_gray[
        coords_inside_cell[:, 0], coords_inside_cell[:, 1]
    ]
    points_inside_cell_2 = fluo2_gray[
        coords_inside_cell[:, 0], coords_inside_cell[:, 1]
    ]

    style = "dark_background" if dark_mode else "default"
    text_kwargs = {"color": "white"} if dark_mode else {}
    margin_width = 20
    margin_height = 20

    with plt.style.context(style):
        fig = plt.figure(figsize=(6, 6))

        plt.scatter(u1_shifted, u2_shifted, s=5, label="Points in cell")
        plt.scatter([0], [0], color="red", s=100, label="Centroid (0,0)")
        plt.axis("equal")

        x_vals = [val[1] for val in U_shifted]
        y_vals = [val[0] for val in U_shifted]
        plt.scatter(
            x_vals,
            y_vals,
            c=points_inside_cell_1,
            cmap="Blues",
            marker="o",
            s=20,
            alpha=0.8,
            label="Fluo1 intensity",
        )
        plt.scatter(
            x_vals,
            y_vals,
            c=points_inside_cell_2,
            cmap="Reds",
            marker="o",
            s=20,
            alpha=0.8,
            label="Fluo2 intensity",
        )

        min_u1_shifted = float(np.min(u1_shifted))
        max_u1_shifted = float(np.max(u1_shifted))
        min_u2_shifted = float(np.min(u2_shifted))
        max_u2_shifted = float(np.max(u2_shifted))
        x_min = min_u1_shifted - margin_width
        x_max = max_u1_shifted + margin_width
        y_min = min_u2_shifted - margin_height
        y_max = max_u2_shifted + margin_height
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        stats_fluo1 = _compute_replot_stats(points_inside_cell_1)
        stats_fluo2 = _compute_replot_stats(points_inside_cell_2)

        def draw_stats(x_pos: float, label: str, stats: dict[str, float]) -> None:
            plt.text(
                x_pos,
                0.35,
                f"{label} SD: {stats['sd']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )
            plt.text(
                x_pos,
                0.30,
                f"{label} CV: {stats['cv']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )
            plt.text(
                x_pos,
                0.25,
                f"{label} Loc: {stats['localization']:.4g}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )
            plt.text(
                x_pos,
                0.20,
                f"{label} Med: {stats['median']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )
            plt.text(
                x_pos,
                0.15,
                f"{label} Mean: {stats['mean']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )
            plt.text(
                x_pos,
                0.10,
                f"{label} NMed: {stats['normalized_median']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )
            plt.text(
                x_pos,
                0.05,
                f"{label} NMean: {stats['normalized_mean']:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                **text_kwargs,
            )

        draw_stats(0.28, "F1", stats_fluo1)
        draw_stats(0.72, "F2", stats_fluo2)

        x_for_fit = np.linspace(min_u1_shifted, max_u1_shifted, 1000)
        theta = _poly_fit(U_shifted, degree=degree)
        y_for_fit = np.polyval(theta, x_for_fit)
        plt.plot(x_for_fit, y_for_fit, color="red", label="Poly fit")

        plt.scatter(
            u1_contour_shifted,
            u2_contour_shifted,
            color="lime",
            s=20,
            label="Contour",
        )

        plt.tick_params(direction="in")
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

    return buf.getvalue()


def _find_path_vector(
    image_fluo_raw: bytes,
    contour_raw: bytes,
    degree: int,
) -> list[tuple[float, float]]:
    image_fluo = cv2.imdecode(np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR)
    if image_fluo is None:
        raise ValueError("Failed to decode image")
    image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)

    contour = pickle.loads(contour_raw)
    contour_array = np.asarray(contour)
    if contour_array.ndim == 3 and contour_array.shape[1] == 1:
        contour_points = contour_array[:, 0, :].tolist()
        contour_for_mask = contour_array.astype(np.int32)
    elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
        contour_points = contour_array.tolist()
        contour_for_mask = contour_array.reshape(-1, 1, 2).astype(np.int32)
    else:
        raise ValueError("Invalid contour format")

    mask = np.zeros_like(image_fluo_gray)
    cv2.fillPoly(mask, [contour_for_mask], 255)

    coords_inside_cell = np.column_stack(np.where(mask))
    if coords_inside_cell.size == 0:
        raise ValueError("No points inside contour")
    points_inside_cell = image_fluo_gray[
        coords_inside_cell[:, 0], coords_inside_cell[:, 1]
    ]

    X = np.array(
        [
            [i[1] for i in coords_inside_cell],
            [i[0] for i in coords_inside_cell],
        ]
    )

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
        coords_inside_cell.tolist(),
    )

    theta = _poly_fit(U, degree=degree)

    raw_points: list[tuple[float, float]] = []
    for u1_val, u2_val, intensity in zip(u1, u2, points_inside_cell):
        _dist, min_point = _find_minimum_distance_point(
            theta, float(u1_val), float(u2_val), float(min_u1), float(max_u1)
        )
        raw_points.append((min_point[0], float(intensity)))

    if not raw_points:
        return []

    raw_points.sort(key=lambda pair: pair[0])

    split_num = 35
    delta_l = (max(u1) - min(u1)) / split_num if split_num > 0 else 0
    if delta_l == 0:
        return raw_points

    first_point = raw_points[0]
    last_point = raw_points[-1]
    path: list[tuple[float, float]] = [first_point]
    for idx in range(1, int(split_num)):
        x_0 = min(u1) + idx * delta_l
        x_1 = min(u1) + (idx + 1) * delta_l
        points = [point for point in raw_points if x_0 <= point[0] <= x_1]
        if not points:
            continue
        point = max(points, key=lambda pair: pair[1])
        path.append(point)
    path.append(last_point)

    return path


def _generate_heatmap_image(path: Sequence[Sequence[float]]) -> bytes:
    if not path:
        raise ValueError("Heatmap path is empty")
    values = [val[1] for val in path]
    data = np.array(values).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    cax = ax.imshow(data, cmap="inferno", interpolation="nearest", aspect="auto")
    fig.colorbar(cax, ax=ax)
    ax.set_ylabel("Relative position")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def _build_map256_normalized(
    image_fluo_raw: bytes,
    contour_raw: bytes,
    degree: int,
) -> np.ndarray:
    image_fluo = cv2.imdecode(np.frombuffer(image_fluo_raw, np.uint8), cv2.IMREAD_COLOR)
    if image_fluo is None:
        raise ValueError("Failed to decode image")
    image_fluo_gray = cv2.cvtColor(image_fluo, cv2.COLOR_BGR2GRAY)
    image_fluo_gray = _subtract_background(image_fluo_gray)

    contour = pickle.loads(contour_raw)
    contour_array = np.asarray(contour)
    if contour_array.ndim == 3 and contour_array.shape[1] == 1:
        contour_points = contour_array[:, 0, :].tolist()
        contour_for_mask = contour_array.astype(np.int32)
    elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
        contour_points = contour_array.tolist()
        contour_for_mask = contour_array.reshape(-1, 1, 2).astype(np.int32)
    else:
        raise ValueError("Invalid contour format")

    mask = np.zeros_like(image_fluo_gray)
    cv2.fillPoly(mask, [contour_for_mask], 255)
    coords_inside_cell = np.column_stack(np.where(mask))
    if coords_inside_cell.size == 0:
        raise ValueError("No points inside contour")
    points_inside_cell = image_fluo_gray[
        coords_inside_cell[:, 0], coords_inside_cell[:, 1]
    ]

    X = np.array(
        [
            [i[1] for i in coords_inside_cell],
            [i[0] for i in coords_inside_cell],
        ]
    )
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
        coords_inside_cell.tolist(),
    )

    theta = _poly_fit(U, degree=degree)
    xs, arc_lengths = _build_arc_length_lookup(theta, float(min_u1), float(max_u1))

    raw_points: list[tuple[float, float, float]] = []
    for u1_val, u2_val, intensity in zip(u1, u2, points_inside_cell):
        dist, min_point = _find_minimum_distance_point(
            theta, float(u1_val), float(u2_val), float(min_u1), float(max_u1)
        )
        sign = 1 if float(u2_val) > float(min_point[1]) else -1
        arc_length = float(np.interp(min_point[0], xs, arc_lengths))
        raw_points.append((arc_length, dist * sign, float(intensity)))

    if not raw_points:
        raise ValueError("No points inside contour")

    ps = np.array([point[0] for point in raw_points], dtype=float)
    dists = np.array([point[1] for point in raw_points], dtype=float)
    gs = np.array([point[2] for point in raw_points], dtype=float)

    min_p, max_p = float(ps.min()), float(ps.max())
    min_dist, max_dist = float(dists.min()), float(dists.max())

    width = max(1, int(np.ceil(max_p - min_p)) + 1)
    height = max(1, int(np.ceil(max_dist - min_dist)) + 1)

    lowest_intensity = int(np.min(points_inside_cell))
    high_res_image = np.full((height, width), lowest_intensity, dtype=np.uint8)

    for p_val, dist_val, g_val in zip(ps, dists, gs):
        x = int(p_val - min_p) if width > 1 else 0
        y = int(dist_val - min_dist) if height > 1 else 0
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        cv2.circle(high_res_image, (x, y), 1, int(np.clip(g_val, 0, 255)), -1)

    resized = cv2.resize(high_res_image, (1024, 256), interpolation=cv2.INTER_NEAREST)
    resized = _flip_image_if_needed(resized)
    normalized = _normalize_grayscale_to_uint8(resized)
    return normalized


def _generate_map256_image(
    image_fluo_raw: bytes,
    contour_raw: bytes,
    degree: int,
) -> bytes:
    normalized = _build_map256_normalized(image_fluo_raw, contour_raw, degree)
    return _encode_image(normalized)


def _generate_map256_jet_image(
    image_fluo_raw: bytes,
    contour_raw: bytes,
    degree: int,
) -> bytes:
    normalized = _build_map256_normalized(image_fluo_raw, contour_raw, degree)
    jet = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return _encode_image(jet)


def get_cell_image(
    db_name: str,
    cell_id: str,
    image_type: Literal["ph", "fluo1", "fluo2"],
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    gain: float = 1.0,
    fluo_color: str | None = None,
) -> bytes:
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Gain must be greater than 0")
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "ph": "img_ph",
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        columns = [cells.c[column_name]]
        if draw_contour:
            columns.append(cells.c.contour)
        stmt = (
            select(*columns)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        image_bytes = bytes(row[0])
        needs_processing = (
            draw_contour or draw_scale_bar or image_type in ("fluo1", "fluo2")
        )
        if not needs_processing:
            return image_bytes

        image = _decode_image(image_bytes)
        if image_type in ("fluo1", "fluo2"):
            if gain != 1:
                gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                scaled = gray.astype(np.float32) * float(gain)
                image = np.clip(scaled, 0, 255).astype(np.uint8)
            image = _colorize_fluo_image(image, image_type, fluo_color=fluo_color)
        if draw_contour:
            contour_raw = row[1] if len(row) > 1 else None
            if contour_raw is None:
                raise LookupError("Cell contour not found")
            image = _draw_contour(image, contour_raw)
        if draw_scale_bar:
            image = _draw_scale_bar_with_centered_text(image)
        return _encode_image(image)
    finally:
        session.close()


def _render_cell_image_from_blob(
    image_blob: bytes,
    image_type: str,
    contour_raw: bytes | None = None,
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    gain: float = 1.0,
    fluo_color: str | None = None,
) -> bytes:
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Gain must be greater than 0")
    if image_type not in ("ph", "fluo1", "fluo2"):
        raise ValueError("Invalid image_type")
    image = _decode_image(image_blob)
    if image_type in ("fluo1", "fluo2"):
        if gain != 1:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scaled = gray.astype(np.float32) * float(gain)
            image = np.clip(scaled, 0, 255).astype(np.uint8)
        image = _colorize_fluo_image(image, image_type, fluo_color=fluo_color)
    if draw_contour and contour_raw is not None:
        image = _draw_contour(image, contour_raw)
    if draw_scale_bar:
        image = _draw_scale_bar_with_centered_text(image)
    return _encode_image(image)


def get_cell_image_optical_boost(
    db_name: str,
    cell_id: str,
    image_type: Literal["fluo1", "fluo2"],
    draw_contour: bool = False,
    draw_scale_bar: bool = False,
    fluo_color: str | None = None,
) -> bytes:
    if image_type not in ("fluo1", "fluo2"):
        raise ValueError("Optical boost only supports fluo1 or fluo2")
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        columns = [cells.c[column_name]]
        if draw_contour:
            columns.append(cells.c.contour)
        stmt = (
            select(*columns)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        image = _decode_image(bytes(row[0]))
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = _normalize_grayscale_to_uint8(gray)
        boosted = _colorize_fluo_image(normalized, image_type, fluo_color=fluo_color)
        if draw_contour:
            contour_raw = row[1] if len(row) > 1 else None
            if contour_raw is None:
                raise LookupError("Cell contour not found")
            boosted = _draw_contour(boosted, contour_raw)
        if draw_scale_bar:
            boosted = _draw_scale_bar_with_centered_text(boosted)
        return _encode_image(boosted)
    finally:
        session.close()


def _render_cell_overlay_from_blobs(
    ph_raw: bytes | None,
    fluo1_raw: bytes | None,
    fluo2_raw: bytes | None,
    contour_raw: bytes | None,
    draw_scale_bar: bool = False,
    overlay_mode: Literal["ph", "fluo", "raw"] = "ph",
    scale: float = 1.0,
    fluo1_color: str | None = None,
    fluo2_color: str | None = None,
    image_format: Literal["png", "jpeg", "jpg"] = "png",
    jpeg_quality: int = 80,
) -> bytes:
    if overlay_mode not in ("ph", "fluo", "raw"):
        raise ValueError("Invalid overlay_mode")
    if fluo1_raw is None:
        raise LookupError("Cell overlay data not found")
    if overlay_mode == "ph" and contour_raw is None:
        raise LookupError("Cell overlay data not found")

    fluo1_image = _decode_image(bytes(fluo1_raw))
    fluo2_image = _decode_image(bytes(fluo2_raw)) if fluo2_raw is not None else None
    ph_image = None
    if overlay_mode in ("ph", "raw"):
        if ph_raw is None:
            raise LookupError("Cell overlay data not found")
        ph_image = _decode_image(bytes(ph_raw))
        if ph_image.shape[:2] != fluo1_image.shape[:2]:
            raise ValueError("Overlay image sizes do not match")
        if fluo2_image is not None and ph_image.shape[:2] != fluo2_image.shape[:2]:
            raise ValueError("Overlay image sizes do not match")
        overlay = ph_image.copy()
    else:
        overlay = np.zeros_like(fluo1_image)
        if fluo2_image is not None and fluo1_image.shape[:2] != fluo2_image.shape[:2]:
            raise ValueError("Overlay image sizes do not match")

    if overlay_mode == "ph":
        contour = pickle.loads(bytes(contour_raw))
        contour_array = np.asarray(contour)
        if contour_array.ndim == 3 and contour_array.shape[1] == 1:
            contour_np = contour_array.astype(np.int32)
        elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
            contour_np = contour_array.reshape(-1, 1, 2).astype(np.int32)
        else:
            raise ValueError("Invalid contour format")

        mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour_np], 255)
        mask_bool = mask > 0
        if not np.any(mask_bool):
            raise ValueError("No pixels inside contour")
    else:
        mask_bool = np.ones(overlay.shape[:2], dtype=bool)

    fluo1_gray = cv2.cvtColor(fluo1_image, cv2.COLOR_BGR2GRAY)
    fluo2_gray = (
        cv2.cvtColor(fluo2_image, cv2.COLOR_BGR2GRAY) if fluo2_image is not None else None
    )

    min1 = float(fluo1_gray[mask_bool].min())
    max1 = float(fluo1_gray[mask_bool].max())
    min2 = float(fluo2_gray[mask_bool].min()) if fluo2_gray is not None else 0.0
    max2 = float(fluo2_gray[mask_bool].max()) if fluo2_gray is not None else 0.0

    range1 = max1 - min1
    range2 = max2 - min2 if fluo2_gray is not None else 0.0
    if range1 > 0:
        norm1 = ((fluo1_gray.astype(np.float32) - min1) / range1) * 255.0
    else:
        norm1 = np.zeros_like(fluo1_gray, dtype=np.float32)
    if fluo2_gray is not None:
        if range2 > 0:
            norm2 = ((fluo2_gray.astype(np.float32) - min2) / range2) * 255.0
        else:
            norm2 = np.zeros_like(fluo2_gray, dtype=np.float32)
    else:
        norm2 = None

    norm1 = np.clip(norm1, 0, 255).astype(np.uint8)
    if norm2 is not None:
        norm2 = np.clip(norm2, 0, 255).astype(np.uint8)

    color1 = _resolve_fluo_color(fluo1_color, "fluo1")
    _apply_fluo_overlay(overlay, norm1, mask_bool, color1)
    if norm2 is not None:
        color2 = _resolve_fluo_color(fluo2_color, "fluo2")
        _apply_fluo_overlay(overlay, norm2, mask_bool, color2)

    if draw_scale_bar:
        overlay = _draw_scale_bar_with_centered_text(overlay)

    if scale <= 0 or scale > 1:
        raise ValueError("Invalid scale")
    if scale < 1:
        width = max(1, int(overlay.shape[1] * scale))
        height = max(1, int(overlay.shape[0] * scale))
        overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_AREA)

    return _encode_image_with_format(
        overlay,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def get_cell_overlay(
    db_name: str,
    cell_id: str,
    draw_scale_bar: bool = False,
    overlay_mode: Literal["ph", "fluo", "raw"] = "ph",
    scale: float = 1.0,
    fluo1_color: str | None = None,
    fluo2_color: str | None = None,
    image_format: Literal["png", "jpeg", "jpg"] = "png",
    jpeg_quality: int = 80,
) -> bytes:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(
                cells.c.img_ph,
                cells.c.img_fluo1,
                cells.c.img_fluo2,
                cells.c.contour,
            )
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None:
            raise LookupError("Cell overlay data not found")

        ph_raw, fluo1_raw, fluo2_raw, contour_raw = row
        return _render_cell_overlay_from_blobs(
            bytes(ph_raw) if ph_raw is not None else None,
            bytes(fluo1_raw) if fluo1_raw is not None else None,
            bytes(fluo2_raw) if fluo2_raw is not None else None,
            bytes(contour_raw) if contour_raw is not None else None,
            draw_scale_bar=draw_scale_bar,
            overlay_mode=overlay_mode,
            scale=scale,
            fluo1_color=fluo1_color,
            fluo2_color=fluo2_color,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
    finally:
        session.close()


def get_cell_overlay_zip(
    db_name: str,
    draw_scale_bar: bool = False,
    overlay_mode: Literal["ph", "fluo", "raw"] = "ph",
    scale: float = 1.0,
    fluo1_color: str | None = None,
    fluo2_color: str | None = None,
    image_format: Literal["png", "jpeg", "jpg"] = "png",
    jpeg_quality: int = 80,
) -> bytes:
    if overlay_mode not in ("ph", "fluo", "raw"):
        raise ValueError("Invalid overlay_mode")
    if image_format not in ("png", "jpeg", "jpg"):
        raise ValueError("Invalid image format")
    if scale <= 0 or scale > 1:
        raise ValueError("Invalid scale")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(
                cells.c.cell_id,
                cells.c.img_ph,
                cells.c.img_fluo1,
                cells.c.img_fluo2,
                cells.c.contour,
            )
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )
        result = session.execute(stmt)

        extension = "jpg" if image_format.lower() in ("jpeg", "jpg") else "png"
        manifest_entries: list[dict[str, str]] = []
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, (cell_id, ph_raw, fluo1_raw, fluo2_raw, contour_raw) in enumerate(
                result
            ):
                if cell_id is None:
                    continue
                cell_id_str = str(cell_id)
                try:
                    image_bytes = _render_cell_overlay_from_blobs(
                        bytes(ph_raw) if ph_raw is not None else None,
                        bytes(fluo1_raw) if fluo1_raw is not None else None,
                        bytes(fluo2_raw) if fluo2_raw is not None else None,
                        bytes(contour_raw) if contour_raw is not None else None,
                        draw_scale_bar=draw_scale_bar,
                        overlay_mode=overlay_mode,
                        scale=scale,
                        fluo1_color=fluo1_color,
                        fluo2_color=fluo2_color,
                        image_format=image_format,
                        jpeg_quality=jpeg_quality,
                    )
                except Exception:
                    continue

                filename = (
                    f"images/{index:06d}_{_safe_cell_filename(cell_id_str)}.{extension}"
                )
                archive.writestr(filename, image_bytes)
                manifest_entries.append(
                    {
                        "cell_id": cell_id_str,
                        "file": filename,
                    }
                )
            archive.writestr(
                "manifest.json",
                json.dumps({"cells": manifest_entries}, ensure_ascii=True),
            )
        buffer.seek(0)
        return buffer.getvalue()
    finally:
        session.close()


def get_cell_replot(
    db_name: str,
    cell_id: str,
    image_type: Literal["fluo1", "fluo2", "ph", "overlay"] = "fluo1",
    degree: int = 4,
    dark_mode: bool = False,
) -> bytes:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        if image_type == "overlay":
            stmt = (
                select(cells.c.img_fluo1, cells.c.img_fluo2, cells.c.contour)
                .where(cells.c.cell_id == cell_id)
                .limit(1)
            )
            row = session.execute(stmt).first()
            if row is None or row[0] is None or row[1] is None:
                raise LookupError("Cell image not found")
            if row[2] is None:
                raise LookupError("Cell contour not found")
            return _generate_replot_overlay_image(
                bytes(row[0]),
                bytes(row[1]),
                bytes(row[2]),
                degree=degree,
                dark_mode=dark_mode,
            )
        column_map = {
            "ph": "img_ph",
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        stmt = (
            select(cells.c[column_name], cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        if row[1] is None:
            raise LookupError("Cell contour not found")
        image_bytes = bytes(row[0])
        contour_raw = bytes(row[1])
        return _generate_replot_image(
            image_bytes,
            contour_raw,
            degree=degree,
            dark_mode=dark_mode,
        )
    finally:
        session.close()


def get_cell_heatmap(
    db_name: str,
    cell_id: str,
    image_type: Literal["fluo1", "fluo2"] = "fluo1",
    degree: int = 4,
) -> bytes:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        stmt = (
            select(cells.c[column_name], cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        if row[1] is None:
            raise LookupError("Cell contour not found")
        image_bytes = bytes(row[0])
        contour_raw = bytes(row[1])
        path = _find_path_vector(image_bytes, contour_raw, degree)
        return _generate_heatmap_image(path)
    finally:
        session.close()


def get_cell_map256(
    db_name: str,
    cell_id: str,
    image_type: Literal["fluo1", "fluo2"] = "fluo1",
    degree: int = 4,
) -> bytes:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        stmt = (
            select(cells.c[column_name], cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        if row[1] is None:
            raise LookupError("Cell contour not found")
        image_bytes = bytes(row[0])
        contour_raw = bytes(row[1])
        return _generate_map256_image(image_bytes, contour_raw, degree)
    finally:
        session.close()


def get_cell_map256_jet(
    db_name: str,
    cell_id: str,
    image_type: Literal["fluo1", "fluo2"] = "fluo1",
    degree: int = 4,
) -> bytes:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        stmt = (
            select(cells.c[column_name], cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        if row[1] is None:
            raise LookupError("Cell contour not found")
        image_bytes = bytes(row[0])
        contour_raw = bytes(row[1])
        return _generate_map256_jet_image(image_bytes, contour_raw, degree)
    finally:
        session.close()


def get_cell_intensity_distribution(
    db_name: str,
    cell_id: str,
    image_type: Literal["ph", "fluo1", "fluo2"] = "fluo1",
) -> bytes:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "ph": "img_ph",
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")
        stmt = (
            select(cells.c[column_name], cells.c.contour)
            .where(cells.c.cell_id == cell_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None or row[0] is None:
            raise LookupError("Cell image not found")
        if row[1] is None:
            raise LookupError("Cell contour not found")
        image = cv2.imdecode(np.frombuffer(bytes(row[0]), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        if image.ndim == 2:
            image_gray = image
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour_raw = bytes(row[1])
        contour = pickle.loads(contour_raw)
        contour_array = np.asarray(contour)
        if contour_array.ndim == 3 and contour_array.shape[1] == 1:
            contour_np = contour_array.astype(np.int32)
        elif contour_array.ndim == 2 and contour_array.shape[1] == 2:
            contour_np = contour_array.reshape(-1, 1, 2).astype(np.int32)
        else:
            raise ValueError("Invalid contour format")
        mask = np.zeros(image_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour_np], 255)
        points = image_gray[mask.astype(bool)]
        if points.size == 0:
            raise ValueError("No pixels inside contour")
        values = np.clip(points, 0, 255).astype(np.uint8)
        counts, _ = np.histogram(values, bins=np.arange(257))

        fig, ax = plt.subplots(figsize=(4, 4), dpi=500)
        ax.bar(np.arange(256), counts, width=1.0, color="#4a5568")
        ax.set_xlim(-0.5, 255.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("fluo intensity")
        ax.set_ylabel("count")
        ax.set_title("8-bit intensity")
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    finally:
        session.close()


def get_annotation_zip(
    db_name: str,
    image_type: Literal["ph", "fluo1", "fluo2"] = "ph",
    raw: bool = False,
    downscale: float | None = None,
) -> bytes:
    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")
        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        column_map = {
            "ph": "img_ph",
            "fluo1": "img_fluo1",
            "fluo2": "img_fluo2",
        }
        column_name = column_map.get(image_type)
        if column_name is None:
            raise ValueError("Invalid image_type")

        effective_downscale = 1.0
        if not raw and image_type == "ph":
            if downscale is None:
                effective_downscale = ANNOTATION_DOWNSCALE
            else:
                try:
                    effective_downscale = float(downscale)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid downscale value") from exc
            if effective_downscale <= 0 or effective_downscale > 1.0:
                raise ValueError("Downscale must be between 0 and 1")
        stmt = (
            select(
                cells.c.cell_id,
                cells.c.manual_label,
                cells.c[column_name],
                cells.c.contour,
            )
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )
        result = session.execute(stmt)

        manifest_entries: list[dict[str, str]] = []
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for cell_id, manual_label, image_blob, contour_blob in result:
                if cell_id is None or image_blob is None or contour_blob is None:
                    continue
                cell_id_str = str(cell_id)
                filename = f"images/{_safe_cell_filename(cell_id_str)}.png"
                try:
                    image = _decode_image(bytes(image_blob))
                    if image_type in ("fluo1", "fluo2"):
                        image = _colorize_fluo_image(image, image_type)
                    image = _draw_contour(
                        image,
                        bytes(contour_blob),
                        thickness=ANNOTATION_CONTOUR_THICKNESS,
                    )
                    if effective_downscale < 1.0:
                        height, width = image.shape[:2]
                        new_width = max(int(width * effective_downscale), 1)
                        new_height = max(int(height * effective_downscale), 1)
                        if new_width != width or new_height != height:
                            image = cv2.resize(
                                image,
                                (new_width, new_height),
                                interpolation=cv2.INTER_AREA,
                            )
                    image = _pad_to_square(image)
                    image_bytes = _encode_image(image)
                except Exception:
                    continue
                archive.writestr(filename, image_bytes)
                manifest_entries.append(
                    {
                        "cell_id": cell_id_str,
                        "label": _normalize_annotation_label(manual_label),
                        "file": filename,
                    }
                )
            archive.writestr(
                "manifest.json",
                json.dumps({"cells": manifest_entries}, ensure_ascii=True),
            )
        buffer.seek(0)
        return buffer.getvalue()
    finally:
        session.close()


def get_cells_fast_bundle(
    db_name: str,
    label: str | None = None,
    draw_contour: bool = True,
    draw_scale_bar: bool = True,
    fluo1_color: str | None = None,
    fluo2_color: str | None = None,
    gain: float = 1.0,
) -> bytes:
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Gain must be greater than 0")

    resolved_fluo1 = _resolve_fluo_color(fluo1_color, "fluo1")
    resolved_fluo2 = _resolve_fluo_color(fluo2_color, "fluo2")
    label_value = label.strip() if label is not None else None
    if label is not None and not label_value:
        raise ValueError("Label is required")

    session = get_database_session(db_name)
    try:
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Database session is not bound")

        metadata = MetaData()
        cells = Table("cells", metadata, autoload_with=bind)
        stmt = (
            select(
                cells.c.cell_id,
                cells.c.manual_label,
                cells.c.img_ph,
                cells.c.img_fluo1,
                cells.c.img_fluo2,
                cells.c.contour,
            )
            .where(cells.c.cell_id.is_not(None))
            .order_by(cells.c.cell_id)
        )
        if label_value is not None:
            filters = [cells.c.manual_label == label_value]
            if label_value.isdigit():
                filters.append(cells.c.manual_label == int(label_value))
            stmt = stmt.where(or_(*filters))

        result = session.execute(stmt)
        bundle = io.BytesIO()
        manifest_cells: list[dict[str, object]] = []

        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, (
                cell_id,
                manual_label,
                ph_blob,
                fluo1_blob,
                fluo2_blob,
                contour_blob,
            ) in enumerate(result):
                if cell_id is None:
                    continue

                cell_id_str = str(cell_id)
                contour_raw = bytes(contour_blob) if contour_blob is not None else None
                safe_prefix = f"{index:06d}_{_safe_cell_filename(cell_id_str)}"

                files: dict[str, str] = {}
                missing = {
                    "ph": ph_blob is None,
                    "fluo1": fluo1_blob is None,
                    "fluo2": fluo2_blob is None,
                }

                channel_specs: tuple[tuple[str, object, str | None], ...] = (
                    ("ph", ph_blob, None),
                    ("fluo1", fluo1_blob, resolved_fluo1),
                    ("fluo2", fluo2_blob, resolved_fluo2),
                )
                for channel_name, blob_value, channel_color in channel_specs:
                    if blob_value is None:
                        continue
                    rendered = _render_cell_image_from_blob(
                        bytes(blob_value),
                        channel_name,
                        contour_raw=contour_raw,
                        draw_contour=draw_contour,
                        draw_scale_bar=draw_scale_bar,
                        gain=gain if channel_name in ("fluo1", "fluo2") else 1.0,
                        fluo_color=channel_color,
                    )
                    image_path = f"images/{safe_prefix}_{channel_name}.png"
                    archive.writestr(image_path, rendered)
                    files[channel_name] = image_path

                manifest_cells.append(
                    {
                        "cell_id": cell_id_str,
                        "manual_label": _normalize_fast_manual_label(manual_label),
                        "files": files,
                        "missing": missing,
                    }
                )

            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "cells": manifest_cells,
                        "settings": {
                            "draw_contour": draw_contour,
                            "draw_scale_bar": draw_scale_bar,
                            "fluo1_color": resolved_fluo1,
                            "fluo2_color": resolved_fluo2,
                            "gain": gain,
                        },
                    },
                    ensure_ascii=True,
                ),
            )

        bundle.seek(0)
        return bundle.getvalue()
    finally:
        session.close()


class DatabaseManagerCrud:
    DATABASES_DIR = DATABASES_DIR
    DOWNLOAD_CHUNK_SIZE = DOWNLOAD_CHUNK_SIZE
    ANNOTATION_DOWNSCALE = ANNOTATION_DOWNSCALE
    ANNOTATION_CONTOUR_THICKNESS = ANNOTATION_CONTOUR_THICKNESS

    @classmethod
    def sanitize_db_name(cls, db_name: str) -> str:
        return sanitize_db_name(db_name)

    @classmethod
    def resolve_database_path(cls, db_name: str) -> Path:
        return resolve_database_path(db_name)

    @classmethod
    def read_database_chunks(
        cls, path: Path, chunk_size: int = DOWNLOAD_CHUNK_SIZE
    ) -> AsyncIterator[bytes]:
        return read_database_chunks(path, chunk_size)

    @classmethod
    async def delete_database(cls, db_name: str) -> str:
        return await delete_database(db_name)

    @classmethod
    def get_database_session(cls, db_name: str) -> Session:
        return get_database_session(db_name)

    @classmethod
    def migrate_database(cls, db_name: str) -> None:
        migrate_database(db_name)

    @classmethod
    async def list_databases(cls) -> list[str]:
        return await list_databases()

    @classmethod
    def get_cell_ids(cls, db_name: str) -> list[str]:
        return get_cell_ids(db_name)

    @classmethod
    def get_cell_label(cls, db_name: str, cell_id: str) -> str:
        return get_cell_label(db_name, cell_id)

    @classmethod
    def update_cell_label(cls, db_name: str, cell_id: str, label: str) -> str:
        return update_cell_label(db_name, cell_id, label)

    @classmethod
    def get_cell_contour(cls, db_name: str, cell_id: str) -> list[list[float]]:
        return get_cell_contour(db_name, cell_id)

    @classmethod
    def apply_elastic_contour(
        cls, db_name: str, cell_id: str, delta: float
    ) -> list[list[float]]:
        return apply_elastic_contour(db_name, cell_id, delta)

    @classmethod
    def apply_elastic_contour_bulk(
        cls, db_name: str, delta: int, label: str | None = None
    ) -> dict:
        return apply_elastic_contour_bulk(db_name, delta, label)

    @classmethod
    def get_cell_ids_by_label(cls, db_name: str, label: str) -> list[str]:
        return get_cell_ids_by_label(db_name, label)

    @classmethod
    def get_manual_labels(cls, db_name: str) -> list[str]:
        return get_manual_labels(db_name)

    @classmethod
    def build_map256_normalized(
        cls, image_fluo_raw: bytes, contour_raw: bytes, degree: int
    ) -> np.ndarray:
        return _build_map256_normalized(image_fluo_raw, contour_raw, degree)

    @classmethod
    def get_cell_image(cls, *args, **kwargs) -> bytes:
        return get_cell_image(*args, **kwargs)

    @classmethod
    def get_cell_image_optical_boost(cls, *args, **kwargs) -> bytes:
        return get_cell_image_optical_boost(*args, **kwargs)

    @classmethod
    def get_cell_overlay(cls, *args, **kwargs) -> bytes:
        return get_cell_overlay(*args, **kwargs)

    @classmethod
    def get_cell_overlay_zip(cls, *args, **kwargs) -> bytes:
        return get_cell_overlay_zip(*args, **kwargs)

    @classmethod
    def get_cell_replot(cls, *args, **kwargs) -> bytes:
        return get_cell_replot(*args, **kwargs)

    @classmethod
    def get_cell_heatmap(cls, *args, **kwargs) -> bytes:
        return get_cell_heatmap(*args, **kwargs)

    @classmethod
    def get_cell_map256(cls, *args, **kwargs) -> bytes:
        return get_cell_map256(*args, **kwargs)

    @classmethod
    def get_cell_map256_jet(cls, *args, **kwargs) -> bytes:
        return get_cell_map256_jet(*args, **kwargs)

    @classmethod
    def get_cell_intensity_distribution(cls, *args, **kwargs) -> bytes:
        return get_cell_intensity_distribution(*args, **kwargs)

    @classmethod
    def get_annotation_zip(cls, *args, **kwargs) -> bytes:
        return get_annotation_zip(*args, **kwargs)

    @classmethod
    def get_cells_fast_bundle(cls, *args, **kwargs) -> bytes:
        return get_cells_fast_bundle(*args, **kwargs)
