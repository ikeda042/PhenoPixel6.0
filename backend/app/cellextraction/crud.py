import os
import shutil
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal, Optional

import cv2
import nd2reader
import numpy as np
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlalchemy import BLOB, Column, FLOAT, Integer, String, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import select

from pydantic import BaseModel, Field, ValidationInfo, field_validator


APP_DIR = Path(__file__).resolve().parents[1]
DATABASES_DIR = APP_DIR / "databases"
EXTRACTED_DATA_DIR = APP_DIR / "extracted_data"
TEMPDATA_DIR = APP_DIR / "tempdata"


def _get_temp_dir(ulid: str) -> str:
    return str(TEMPDATA_DIR / f"TempData{ulid}")


def second_pca_variance_from_blob(contour_blob: bytes) -> Optional[float]:
    """
    Deserialize a contour BLOB and return the variance of the second PCA axis
    (smaller eigenvalue). Returns None if the contour is invalid.
    """
    try:
        contour = pickle.loads(contour_blob)
    except Exception:
        return None

    # Accept common contour layouts, including OpenCV-style (N, 1, 2).
    arr = np.asarray(contour)
    if arr.size == 0:
        return None

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if arr.size < 4 or arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr.reshape(-1, 2)
    else:
        return None

    if arr.shape[1] < 2:
        return None
    if arr.shape[1] > 2:
        arr = arr[:, :2]

    if arr.shape[0] < 2:
        return None

    points = arr.astype(float, copy=False)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2):
        return None

    eigvals = np.linalg.eigvalsh(cov)
    return float(max(eigvals[0], 0.0))


def convexity_from_contour(contour: object) -> Optional[float]:
    """
    Compute convexity = hull_perimeter / perimeter from a contour-like object.
    Returns None if the contour is invalid or the perimeter is zero.
    """
    arr = np.asarray(contour)
    if arr.size == 0:
        return None

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if arr.size < 4 or arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr.reshape(-1, 2)
    else:
        return None

    if arr.shape[1] < 2:
        return None
    if arr.shape[1] > 2:
        arr = arr[:, :2]

    if arr.shape[0] < 2:
        return None

    points = arr.astype(float, copy=False)
    diffs = np.diff(points, axis=0)
    perimeter = float(
        np.hypot(diffs[:, 0], diffs[:, 1]).sum()
        + np.hypot(*(points[0] - points[-1]))
    )
    if perimeter == 0.0:
        return None

    unique = np.unique(points, axis=0)
    if unique.shape[0] <= 1:
        return None
    if unique.shape[0] == 2:
        hull = unique
    else:
        order = np.lexsort((unique[:, 1], unique[:, 0]))
        pts = unique[order]

        def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = np.vstack((lower[:-1], upper[:-1]))

    hull_diffs = np.diff(hull, axis=0)
    hull_perimeter = float(
        np.hypot(hull_diffs[:, 0], hull_diffs[:, 1]).sum()
        + np.hypot(*(hull[0] - hull[-1]))
    )
    return hull_perimeter / perimeter


def screen_contour(contour_blob: bytes) -> bool:
    variance = second_pca_variance_from_blob(contour_blob)
    convexity = convexity_from_contour(pickle.loads(contour_blob))
    return (
        variance is not None
        and variance <= 120
        and convexity is not None
        and convexity < 0.85
    )


class FrameSplitConfig(BaseModel):
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    db_name: str

    @field_validator("db_name")
    @classmethod
    def validate_db_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("db_name cannot be empty")
        return value

    @field_validator("frame_end")
    @classmethod
    def validate_range(cls, value: int, info: ValidationInfo) -> int:
        frame_start = info.data.get("frame_start")
        if frame_start is not None and value < frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        return value




def get_ulid() -> str:
    """Return a fake ULID using random digits."""
    # NOTE: This is a placeholder implementation
    return "".join(str(random.randint(0, 9)) for _ in range(16))


Base = declarative_base()


@dataclass
class FrameSplitRange:
    frame_start: int
    frame_end: int
    db_name: str
    db_path: str


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


def get_session(dbname: str) -> Generator[Session, None, None]:
    engine = create_engine(
        f"sqlite:///{dbname}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    session_factory = sessionmaker(engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def create_database(dbname: str) -> Engine:
    db_dir = os.path.dirname(dbname)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{dbname}",
        echo=True,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("PRAGMA journal_mode=DELETE;"))
    return engine


class SyncChores:
    @staticmethod
    def process_image(array) -> np.ndarray:
        """
        画像処理関数：正規化とスケーリングを行う。
        """
        array = array.astype(np.float32)  # Convert to float
        array -= array.min()  # Normalize to 0
        array /= array.max()  # Normalize to 1
        array *= 255  # Scale to 0-255
        return array.astype(np.uint8)

    @staticmethod
    def save_images(num_frames, file_name, num_channels, ulid) -> None:
        """
        画像を保存し、MultipageTIFFとして出力する。
        """
        all_images = []
        for i in range(num_frames):
            if num_channels > 1:
                for j in range(num_channels):
                    all_images.append(
                        Image.open(f"nd2totiff{ulid}/image_{i}_channel_{j}.tif")
                    )
            else:
                all_images.append(Image.open(f"nd2totiff{ulid}/image_{i}.tif"))

        all_images[0].save(
            f"{file_name.split('/')[-1].split('.')[0]}.tif",
            save_all=True,
            append_images=all_images[1:],
        )

        for img in all_images:
            img.close()

    @staticmethod
    def extract_nd2(file_name: str, mode: str, ulid: str, reverse: bool = False) -> int:
        """
        nd2ファイルをMultipageTIFFに変換する。
        """
        temp_dir = f"nd2totiff{ulid}"
        os.makedirs(temp_dir, exist_ok=True)

        with nd2reader.ND2Reader(file_name) as images:
            print(f"Available axes: {images.axes}")
            print(f"Sizes: {images.sizes}")

            images.bundle_axes = "cyx" if "c" in images.axes else "yx"
            images.iter_axes = "v"

            num_channels = images.sizes.get("c", 1)
            print(f"Total images: {len(images)}")
            print(f"Channels: {num_channels}")
            print("##############################################")

            frames_processed = 0
            for n in range(len(images)):
                try:
                    img = images[n]
                except KeyError as e:
                    print(f"KeyError while reading frame {n}: {e}. Stopping extraction.")
                    break
                if num_channels > 1:
                    for channel in range(num_channels):
                        array = np.array(img[channel])
                        array = SyncChores.process_image(array)
                        image = Image.fromarray(array)
                        if reverse:
                            channel = num_channels - channel - 1
                        image.save(f"{temp_dir}/image_{n}_channel_{channel}.tif")
                else:
                    array = np.array(img)
                    array = SyncChores.process_image(array)
                    image = Image.fromarray(array)
                    image.save(f"{temp_dir}/image_{n}.tif")
                frames_processed += 1
            SyncChores.save_images(frames_processed, file_name, num_channels, ulid=ulid)
        SyncChores.cleanup(temp_dir)
        num_tiff = SyncChores.extract_tiff(
            tiff_path=f"./{file_name.split('/')[-1].split('.')[0]}.tif",
            ulid=ulid,
            mode=mode,
            reverse=reverse,
        )
        os.remove(f"./{file_name.split('/')[-1].split('.')[0]}.tif")
        return num_tiff

    @staticmethod
    def extract_tiff(
        tiff_path: str,
        ulid: str,
        mode: Literal[
            "single_layer", "dual_layer", "triple_layer", "quad_layer"
        ] = "dual_layer",
        reverse: bool = False,
    ) -> int:
        temp_dir = _get_temp_dir(ulid)
        os.makedirs(temp_dir, exist_ok=True)
        folders = [
            folder
            for folder in os.listdir(temp_dir)
            if os.path.isdir(os.path.join(temp_dir, folder))
        ]

        layers = {
            "quad_layer": ["Fluo1", "Fluo2", "PH"],  # Fluo3 is ignored
            "triple_layer": ["Fluo1", "Fluo2", "PH"],
            "single_layer": ["PH"],
            "dual_layer": ["Fluo1", "PH"],
        }

        for layer in layers.get(mode, []):
            os.makedirs(f"{temp_dir}/{layer}", exist_ok=True)

        with Image.open(tiff_path) as tiff:
            num_pages = tiff.n_frames
            img_num = 0

            layer_map = {
                "quad_layer": [
                    (0, "PH"),
                    (1, "Fluo1"),
                    (2, "Fluo2"),
                    (3, None),  # skip Fluo3
                ],
                "triple_layer": [(0, "PH"), (1, "Fluo1"), (2, "Fluo2")],
                "single_layer": [(0, "PH")],
                "dual_layer": (
                    [(0, "PH"), (1, "Fluo1")]
                    if not reverse
                    else [(1, "PH"), (0, "Fluo1")]
                ),
            }

            for i in range(num_pages):
                tiff.seek(i)
                layer_idx = i % len(layer_map[mode])
                layer = layer_map[mode][layer_idx][1]
                if layer is not None:
                    filename = f"{temp_dir}/{layer}/{img_num}.tif"
                    print(filename)
                    tiff.save(filename, format="TIFF")
                if layer_idx == len(layer_map[mode]) - 1:
                    img_num += 1

        return num_pages

    @staticmethod
    def cleanup(directory: str) -> None:
        """
        指定されたディレクトリを削除する。
        """
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(directory)

    @staticmethod
    def get_contour_center(contour) -> tuple[int, int]:
        # 輪郭のモーメントを計算して重心を求める
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    @staticmethod
    def crop_contours(image, contours, output_size) -> list[np.ndarray]:
        cropped_images = []
        for contour in contours:
            # 各輪郭の中心座標を取得
            cx, cy = SyncChores.get_contour_center(contour)
            # 　中心座標が画像の中心から離れているものを除外
            if cx > 400 and cx < 2000 and cy > 400 and cy < 2000:
                # 切り抜く範囲を計算
                x1 = max(0, cx - output_size[0] // 2)
                y1 = max(0, cy - output_size[1] // 2)
                x2 = min(image.shape[1], cx + output_size[0] // 2)
                y2 = min(image.shape[0], cy + output_size[1] // 2)
                # 画像を切り抜く
                cropped = image[y1:y2, x1:x2]
                cropped_images.append(cropped)
        return cropped_images

    @staticmethod
    def init(
        input_filename: str,
        num_tiff: int,
        ulid: str,
        param1: int = 130,
        image_size: int = 200,
        mode: Literal[
            "single_layer",
            "dual_layer",
            "triple_layer",
            "quad_layer",
        ] = "dual_layer",
        contour_dir: str | None = None,
    ) -> int:
        temp_dir = _get_temp_dir(ulid)
        print(f"Initializing {temp_dir}")
        print("}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")
        if mode == "quad_layer":
            set_num = 4
            init_folders = ["Fluo1", "Fluo2", "PH", "frames", "app_data"]
        elif mode == "triple_layer":
            set_num = 3
            init_folders = ["Fluo1", "Fluo2", "PH", "frames", "app_data"]
        elif mode == "single_layer":
            set_num = 1
            init_folders = ["PH", "frames", "app_data"]
        else:
            set_num = 2
            init_folders = ["Fluo1", "PH", "frames", "app_data"]

        os.makedirs(temp_dir, exist_ok=True)

        init_folders = [f"{temp_dir}/{d}" for d in init_folders]
        folders = [
            folder
            for folder in os.listdir(f"{temp_dir}")
            if os.path.isdir(os.path.join(".", folder))
        ]
        for i in [i for i in init_folders if i not in folders]:
            try:
                os.mkdir(f"{i}")
            except:
                continue
        # フォルダの作成
        def _ensure_dir(path: str) -> None:
            try:
                os.makedirs(path)
            except Exception as exc:
                print(exc)

        if contour_dir is None:
            stem = os.path.splitext(os.path.basename(input_filename))[0]
            contour_dir = str(EXTRACTED_DATA_DIR / stem)
        _ensure_dir(contour_dir)
        for i in range(num_tiff // set_num):
            frame_dir = f"{temp_dir}/frames/tiff_{i}"
            dirs = [
                frame_dir,
                f"{frame_dir}/Cells",
                f"{frame_dir}/Cells/ph",
                f"{frame_dir}/Cells/fluo1",
            ]
            if mode in ("triple_layer", "quad_layer"):
                dirs.extend(
                    [
                        f"{frame_dir}/Cells/fluo2",
                        f"{frame_dir}/Cells/fluo2_adjusted",
                        f"{frame_dir}/Cells/fluo2_contour",
                    ]
                )
            for path in dirs:
                _ensure_dir(path)
        loop_num = num_tiff // set_num if mode != "single_layer" else num_tiff
        for k in range(loop_num):
            image_ph = cv2.imread(f"{temp_dir}/PH/{k}.tif")
            print(num_tiff)
            if mode == "dual_layer" or mode == "triple_layer" or mode == "quad_layer":
                image_fluo_1 = cv2.imread(f"{temp_dir}/Fluo1/{k}.tif")
            if mode == "triple_layer" or mode == "quad_layer":
                image_fluo_2 = cv2.imread(f"{temp_dir}/Fluo2/{k}.tif")
            img_gray = cv2.cvtColor(image_ph, cv2.COLOR_BGR2GRAY)

            # ２値化を行う
            ret, thresh = cv2.threshold(img_gray, param1, 255, cv2.THRESH_BINARY)
            img_canny = cv2.Canny(thresh, 0, 130)
            contours, hierarchy = cv2.findContours(
                img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # 細胞の面積で絞り込み
            contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours))
            # 中心座標が画像の中心から離れているものを除外
            contours = list(
                filter(
                    lambda x: cv2.moments(x)["m10"] / cv2.moments(x)["m00"] > 400
                    and cv2.moments(x)["m10"] / cv2.moments(x)["m00"] < 1700,
                    contours,
                )
            )
            # do the same for y
            contours = list(
                filter(
                    lambda x: cv2.moments(x)["m01"] / cv2.moments(x)["m00"] > 400
                    and cv2.moments(x)["m01"] / cv2.moments(x)["m00"] < 1700,
                    contours,
                )
            )

            output_size = (image_size, image_size)

            cropped_images_ph = SyncChores.crop_contours(
                image_ph, contours, output_size
            )
            if mode == "triple_layer" or mode == "dual_layer" or mode == "quad_layer":
                cropped_images_fluo_1 = SyncChores.crop_contours(
                    image_fluo_1, contours, output_size
                )
            if mode == "triple_layer" or mode == "quad_layer":
                cropped_images_fluo_2 = SyncChores.crop_contours(
                    image_fluo_2, contours, output_size
                )

            image_ph_copy = image_ph.copy()
            cv2.drawContours(image_ph_copy, contours, -1, (0, 255, 0), 3)
            cv2.imwrite(f"{contour_dir}/{k}.png", image_ph_copy)
            n = 0
            if mode in ("triple_layer", "quad_layer"):
                for ph, fluo1, fluo2 in zip(
                    cropped_images_ph, cropped_images_fluo_1, cropped_images_fluo_2
                ):
                    if (
                        len(ph) == output_size[0]
                        and len(ph[0]) == output_size[1]
                        and len(fluo1) == output_size[0]
                        and len(fluo1[0]) == output_size[1]
                        and len(fluo2) == output_size[0]
                        and len(fluo2[0]) == output_size[1]
                    ):
                        cv2.imwrite(f"{temp_dir}/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        cv2.imwrite(
                            f"{temp_dir}/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1
                        )
                        cv2.imwrite(
                            f"{temp_dir}/frames/tiff_{k}/Cells/fluo2/{n}.png", fluo2
                        )
                        n += 1

            elif mode == "single_layer":
                for ph in cropped_images_ph:
                    if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                        cv2.imwrite(f"{temp_dir}/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        n += 1
            elif mode == "dual_layer":
                for ph, fluo1 in zip(cropped_images_ph, cropped_images_fluo_1):
                    if len(ph) == output_size[0] and len(ph[0]) == output_size[1]:
                        cv2.imwrite(f"{temp_dir}/frames/tiff_{k}/Cells/ph/{n}.png", ph)
                        cv2.imwrite(
                            f"{temp_dir}/frames/tiff_{k}/Cells/fluo1/{n}.png", fluo1
                        )
                        n += 1
        return num_tiff


class ExtractionCrudBase:
    def __init__(
        self,
        nd2_path: str,
        mode: str = "dual_layer",
        param1: int = 130,
        image_size: int = 200,
        reverse_layers: bool = False,
        auto_annotation: bool = False,
        user_id: str | None = None,
        frame_splits: list[FrameSplitConfig] | None = None,
    ) -> None:
        self.nd2_path = nd2_path
        self.nd2_path = self.nd2_path.replace("\\", "/")
        basename = os.path.basename(self.nd2_path)
        base, _ = os.path.splitext(basename)
        self.nd2_stem = base
        self.file_prefix = base.replace(".", "p")
        self.mode = mode
        self.param1 = param1
        self.image_size = image_size
        self.reverse_layers = reverse_layers
        self.auto_annotation = auto_annotation
        self.ulid = get_ulid()
        self.temp_dir = _get_temp_dir(self.ulid)
        self.user_id = user_id
        self.frame_splits = list(frame_splits or [])
        self.contour_dir = str(EXTRACTED_DATA_DIR / self.nd2_stem)

    def load_image(self, path) -> np.ndarray:
        with open(path, "rb") as f:
            data = f.read()
        img_array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    def process_image(
        self, img_ph, img_fluo1=None, img_fluo2=None
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
        img_ph_gray = cv2.cvtColor(img_ph, cv2.COLOR_BGR2GRAY)
        img_fluo1_gray = img_fluo2_gray = None
        if img_fluo1 is not None:
            img_fluo1_gray = cv2.cvtColor(img_fluo1, cv2.COLOR_BGR2GRAY)
        if img_fluo2 is not None:
            img_fluo2_gray = cv2.cvtColor(img_fluo2, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(img_ph_gray, self.param1, 255, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(thresh, 0, 150)
        contours_raw, _ = cv2.findContours(
            img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = list(filter(lambda x: cv2.contourArea(x) >= 300, contours_raw))

        contours = list(
            i
            for i in contours
            if SyncChores.get_contour_center(i)[0] - img_ph.shape[1] // 2 < 3
            and SyncChores.get_contour_center(i)[1] - img_ph.shape[0] // 2 < 3
        )
        contour = contours[0] if contours else None
        return contour, img_ph_gray, img_fluo1_gray, img_fluo2_gray

    def process_cell(
        self,
        session_factory,
        i: int,
        j: int,
        user_id: str | None = None,
    ) -> bool:
        cell_id = f"F{i}C{j}"
        img_ph = self.load_image(
            f"{self.temp_dir}/frames/tiff_{i}/Cells/ph/{j}.png"
        )
        img_fluo1 = img_fluo2 = None
        if self.mode != "single_layer":
            img_fluo1 = self.load_image(
                f"{self.temp_dir}/frames/tiff_{i}/Cells/fluo1/{j}.png"
            )

        contour, img_ph_gray, img_fluo1_gray, _ = self.process_image(img_ph, img_fluo1)
        if contour is None:
            return False

        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        center_x, center_y = SyncChores.get_contour_center(contour)
        if (
            abs(center_x - img_ph.shape[1] // 2) >= 3
            or abs(center_y - img_ph.shape[0] // 2) >= 3
        ):
            return False

        img_ph_data = cv2.imencode(".png", img_ph_gray)[1].tobytes()
        img_fluo1_data = img_fluo2_data = None
        if self.mode != "single_layer":
            img_fluo1_data = cv2.imencode(".png", img_fluo1_gray)[1].tobytes()
        if self.mode in ("triple_layer", "quad_layer"):
            img_fluo2 = self.load_image(
                f"{self.temp_dir}/frames/tiff_{i}/Cells/fluo2/{j}.png"
            )
            img_fluo2_gray = cv2.cvtColor(img_fluo2, cv2.COLOR_BGR2GRAY)
            img_fluo2_data = cv2.imencode(".png", img_fluo2_gray)[1].tobytes()
        cv2.drawContours(img_ph, [contour], -1, (0, 255, 0), 1, cv2.LINE_AA)
        contour_blob = pickle.dumps(contour)
        manual_label: int | str = "N/A"
        if self.auto_annotation:
            manual_label = 1 if screen_contour(contour_blob) else "N/A"
        cell = Cell(
            cell_id=cell_id,
            label_experiment="",
            manual_label=manual_label,
            perimeter=perimeter,
            area=area,
            img_ph=img_ph_data,
            img_fluo1=img_fluo1_data,
            img_fluo2=img_fluo2_data,
            contour=contour_blob,
            center_x=center_x,
            center_y=center_y,
            user_id=user_id,
        )
        with session_factory() as session:
            existing_cell = session.execute(select(Cell).filter_by(cell_id=cell_id))
            existing_cell = existing_cell.scalar()
            if existing_cell is None:
                session.add(cell)
                session.commit()
                return True
        return False

    def _sanitize_db_basename(self, name: str) -> str:
        cleaned = name.strip()
        cleaned = re.sub(r"\.db$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9_\-]", "_", cleaned)
        if not cleaned:
            cleaned = "split"
        stem = re.sub(r"[^A-Za-z0-9_\-]", "_", self.nd2_stem) if self.nd2_stem else ""
        prefix = stem if stem else "nd2file"
        combined = f"{prefix}-{cleaned}"
        combined = re.sub(r"[^A-Za-z0-9_\-]", "_", combined)
        if not combined.lower().endswith(".db"):
            combined = f"{combined}.db"
        return combined

    def _make_unique_basename(
        self, base_name: str, existing: set[str]
    ) -> str:
        candidate = base_name
        counter = 1
        stem, ext = os.path.splitext(base_name)
        while candidate in existing:
            candidate = f"{stem}_{counter}{ext or ''}"
            counter += 1
        existing.add(candidate)
        return candidate

    def _normalize_frame_splits(
        self, frame_count: int, default_db_path: str
    ) -> list[FrameSplitRange]:
        max_frame_index = frame_count - 1 if frame_count > 0 else -1
        normalized: list[FrameSplitRange] = []
        if not self.frame_splits:
            normalized.append(
                FrameSplitRange(
                    frame_start=0,
                    frame_end=max_frame_index,
                    db_name=os.path.basename(default_db_path),
                    db_path=default_db_path,
                )
            )
            return normalized

        existing_names: set[str] = set()
        for cfg in self.frame_splits:
            start = max(0, cfg.frame_start)
            if frame_count > 0 and start > max_frame_index:
                print(
                    f"Split {cfg.frame_start}-{cfg.frame_end} outside available frame range. Skipping."
                )
                continue
            end = cfg.frame_end
            if frame_count > 0:
                end = min(end, max_frame_index)
            if end < start:
                continue
            sanitized = self._sanitize_db_basename(cfg.db_name)
            unique_name = self._make_unique_basename(sanitized, existing_names)
            normalized.append(
                FrameSplitRange(
                    frame_start=start,
                    frame_end=end,
                    db_name=unique_name,
                    db_path=str(DATABASES_DIR / unique_name),
                )
            )

        if not normalized:
            normalized.append(
                FrameSplitRange(
                    frame_start=0,
                    frame_end=max_frame_index,
                    db_name=os.path.basename(default_db_path),
                    db_path=default_db_path,
                )
            )

        normalized.sort(key=lambda split: (split.frame_start, split.frame_end))
        return normalized

    def _reset_database(self, db_path: str) -> None:
        try:
            os.remove(db_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Failed to remove existing database {db_path}: {exc}")
        create_database(db_path)

    def _reset_contour_dir(self) -> None:
        contour_path = Path(self.contour_dir)
        if contour_path.exists():
            if contour_path.is_dir():
                shutil.rmtree(contour_path)
            else:
                contour_path.unlink()

    def _populate_database_range(
        self, db_path: str, frame_start: int, frame_end: int
    ) -> int:
        if frame_end < frame_start:
            return 0

        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        session_factory = sessionmaker(engine, expire_on_commit=False)
        inserted_count = 0

        try:
            for frame_idx in range(frame_start, frame_end + 1):
                cell_path = f"{self.temp_dir}/frames/tiff_{frame_idx}/Cells/ph/"
                if not os.path.exists(cell_path):
                    continue
                cell_indices = sorted(
                    int(path.stem)
                    for path in Path(cell_path).iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ".png"
                    and path.stem.isdigit()
                )
                for j in cell_indices:
                    if self.process_cell(session_factory, frame_idx, j, self.user_id):
                        inserted_count += 1
        finally:
            engine.dispose()
        return inserted_count

    def main(self) -> tuple[int, str, list[dict[str, int | str]]]:
        chores = SyncChores()
        default_db_path = str(DATABASES_DIR / f"{self.file_prefix}.db")

        self._reset_contour_dir()
        num_tiff = chores.extract_nd2(
            self.nd2_path, self.mode, self.ulid, self.reverse_layers
        )

        chores.init(
            f"{self.file_prefix}.nd2",
            num_tiff,
            self.ulid,
            self.param1,
            self.image_size,
            self.mode,
            contour_dir=self.contour_dir,
        )

        iter_n = {
            "triple_layer": num_tiff // 3,
            "quad_layer": num_tiff // 4,
            "single_layer": num_tiff,
            "dual_layer": num_tiff // 2,
        }

        frame_count = iter_n[self.mode]
        splits = self._normalize_frame_splits(frame_count, default_db_path)
        created_databases: list[dict[str, int | str]] = []

        for split in splits:
            self._reset_database(split.db_path)
            contour_count = self._populate_database_range(
                split.db_path, split.frame_start, split.frame_end
            )
            created_databases.append(
                {
                    "frame_start": split.frame_start,
                    "frame_end": split.frame_end,
                    "db_name": split.db_name,
                    "contour_count": contour_count,
                }
            )

        SyncChores.cleanup(self.temp_dir)
        return num_tiff, self.ulid, created_databases

    def get_nd2_filenames(self) -> list[str]:
        upload_dir = "uploaded_files"
        if not os.path.isdir(upload_dir):
            return []
        return [
            i
            for i in os.listdir(upload_dir)
            if i.endswith(".nd2") and not i.endswith("timelapse.nd2")
        ]

    def delete_nd2_file(self, filename: str) -> bool:
        filename = filename.split("/")[-1]
        os.remove(f"uploaded_files/{filename}")
        return True

    def get_ph_contours(
        self, frame_num: int, nd2_stem: str | None = None
    ) -> StreamingResponse:
        contour_dir = (
            str(EXTRACTED_DATA_DIR / nd2_stem) if nd2_stem else self.contour_dir
        )
        filepath = f"{contour_dir}/{frame_num}.png"
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="File not found")
        return StreamingResponse(open(filepath, "rb"), media_type="image/png")

    def get_ph_contours_num(self, nd2_stem: str | None = None) -> int:
        contour_dir = (
            str(EXTRACTED_DATA_DIR / nd2_stem) if nd2_stem else self.contour_dir
        )
        return len(os.listdir(contour_dir))
