import shutil
import numpy as np
import multiprocessing
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import subprocess
import pycolmap

from .utils import load_keypoints, load_matches, logger
from .database import COLMAPDatabase

def satellite_to_local_xy(lon: float, lat: float, scale: float = 1.0):
    """
    Convert satellite image coordinates (lon, lat) to local 3D coordinates.
    For simplicity, assume:
      x = lon * scale, y = lat * scale, and z = 0.0.
    Adjust the scale if your data needs unit conversion.
    """
    x = lon * scale
    y = lat * scale
    z = 0.0
    return x, y, z

def yaw_to_quaternion(yaw_radians: float):
    """
    Convert a yaw angle (rotation about the Z-axis) into a quaternion.
    The quaternion is returned in the COLMAP format: (qw, qx, qy, qz).
    Note: If your yaw is given in degrees, convert it to radians first.
    """
    half = yaw_radians * 0.5
    qw = math.cos(half)
    qx = 0.0
    qy = 0.0
    qz = math.sin(half)
    return (qw, qx, qy, qz)

def set_image_pose_priors(db_path: Path, csv_path: Path, scale: float = 1.0):
    """
    Read the CSV file and update the COLMAP database's images table with the pose priors.
    The CSV must include: File_Name, Pred_Lon, Pred_Lat, Pred_Orien.
    """
    db = COLMAPDatabase.connect(db_path)
    cursor = db.cursor()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  
        for row in reader:
            row = {key.strip(): value for key, value in row.items()}
            name = row['File_Name']
            lon = float(row['Pred_Lon'])
            lat = float(row['Pred_Lat'])
            yaw = float(row['Pred_Orien'])

            x, y, z = satellite_to_local_xy(lon, lat, scale)
            qw, qx, qy, qz = yaw_to_quaternion(yaw)

            cursor.execute("""
                UPDATE images
                SET prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?,
                    prior_tx = ?, prior_ty = ?, prior_tz = ?
                WHERE name = ?
            """, (qw, qx, qy, qz, x, y, z, name))
            logger.info(f"Updated prior pose for image {name}: translation=({x}, {y}, {z}), quaternion=({qw}, {qx}, {qy}, {qz})")

    db.commit()
    db.close()
    logger.info("Image pose priors have been written to the database.")


def create_db_from_model(reference: pycolmap.Reconstruction, database_path: Path) -> Dict[str, int]:
    if database_path.exists():
        logger.warning("Database already exists, deleting the old database.")
        database_path.unlink()
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    for i, cam in reference.cameras.items():
        db.add_camera(
            cam.model.value, cam.width, cam.height, cam.params,
            camera_id=i, prior_focal_length=True
        )
    for i, img in reference.images.items():
        db.add_image(img.name, img.camera_id, image_id=i)
    db.commit()
    db.close()
    return {img.name: i for i, img in reference.images.items()}

def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {name: image_id for name, image_id in db.execute("SELECT name, image_id FROM images;")}
    db.close()
    return images

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("Database already exists, deleting the old database.")
        database_path.unlink()
    logger.info("Creating an empty COLMAP database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    opencv_model_id = 4
    fx, fy = 2038.18, 2034.25
    cx, cy = 1919.89, 1068.13
    k1, k2, p1, p2 = 0.054, -0.135, -0.00076, 0.00111
    opencv_params = np.array([fx, fy, cx, cy, k1, k2, p1, p2], dtype=np.float64)
    db.add_camera(
        model=opencv_model_id,
        width=3840,
        height=2160,
        params=opencv_params,
        prior_focal_length=True,
    )

    db.commit()
    db.close()

def import_images_with_cameras(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=camera_mode,
            image_list=image_list or [],
            options=options,
        )

    db = COLMAPDatabase.connect(database_path)
    cursor = db.cursor()
    cursor.execute("SELECT camera_id FROM cameras WHERE model=4")
    opencv_camera_id = cursor.fetchone()[0]

    cursor.execute("SELECT image_id, name FROM images")
    for image_id, image_name in cursor.fetchall():
        full_path = image_dir / image_name
        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            continue

        img = cv2.imread(str(full_path))
        if img is None:
            logger.warning(f"Failed to read image: {full_path}")
            continue
        h, w = img.shape[:2]

        if (w, h) == (3840, 2160):
            cursor.execute("UPDATE images SET camera_id=? WHERE image_id=?", (opencv_camera_id, image_id))

    db.commit()
    db.close()

def import_features(image_ids: Dict[str, int], database_path: Path, features_path: Path):
    logger.info("Importing features into the database...")
    db = COLMAPDatabase.connect(database_path)
    for image_name, image_id in image_ids.items():
        keypoints = load_keypoints(features_path, image_name)
        keypoints += 0.5  # Convert to COLMAP coordinate system.
        db.add_keypoints(image_id, keypoints)
    db.commit()
    db.close()

def import_matches(
    image_ids: Dict[str, int],
    database_path: Path,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = False,
):
    logger.info("Importing matches into the database...")
    with open(str(pairs_path), "r") as f:
        pairs = [p.strip().split() for p in f.readlines()]
    db = COLMAPDatabase.connect(database_path)
    matched = set()
    for name0, name1 in pairs:
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = load_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}
    db.commit()
    db.close()

def estimation_and_geometric_verification(database_path: Path, pairs_path: Path, verbose: bool = False):
    logger.info("Performing geometric verification...")
    with pycolmap.ostream():
        pycolmap.verify_matches(
            database_path,
            pairs_path,
            options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.1)),
        )

def run_colmap_mapping(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> Optional[pycolmap.Reconstruction]:
    logger.info("Running COLMAP incremental mapping...")
    models_path = sfm_dir / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    options = {"num_threads": min(multiprocessing.cpu_count(), 16)}
    if mapper_options:
        options.update(mapper_options)

    with pycolmap.ostream():
        reconstructions = pycolmap.incremental_mapping(database_path, image_dir, models_path, options=options)

    if not reconstructions:
        logger.error("No reconstructions created!")
        return None

    largest = max(reconstructions.items(), key=lambda kv: kv[1].num_reg_images())
    rec_id, reconstruction = largest

    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.move(str(models_path / str(rec_id) / fname), str(sfm_dir / fname))

    return reconstruction

def run_triangulation(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    logger.info("Running COLMAP triangulation from the reference model...")
    sfm_dir.mkdir(parents=True, exist_ok=True)
    with pycolmap.ostream():
        reconstruction = pycolmap.triangulate_points(reference_model, database_path, image_dir, sfm_dir, options or {})
    return reconstruction

def run_reconstruction(
    sfm_dir: Path,
    image_dir: Path,
    pairs_file: Path,
    features_path: Path,
    matches_path: Path,
    reference_model_path: Optional[Path] = None,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
    pose_prior_csv: Optional[Path] = None,
    pose_prior_scale: float = 1.0,
    pose_prior_confidence: float = 0.0,
) -> Optional[pycolmap.Reconstruction]:
    assert features_path.exists()
    assert pairs_file.exists()
    assert matches_path.exists()
    sfm_dir.mkdir(parents=True, exist_ok=True)
    db_path = sfm_dir / "database.db"

    if mapper_options is None:
        mapper_options = {}

    if reference_model_path is not None:
        logger.info("Triangulation mode using a reference model.")
        assert reference_model_path.exists()
        reference_model = pycolmap.Reconstruction(reference_model_path)
        image_ids = create_db_from_model(reference_model, db_path)
        import_features(image_ids, db_path, features_path)
        import_matches(image_ids, db_path, pairs_file, matches_path, min_match_score, skip_geometric_verification)
        if not skip_geometric_verification:
            estimation_and_geometric_verification(db_path, pairs_file, verbose)
        return run_triangulation(sfm_dir, db_path, image_dir, reference_model, verbose, mapper_options)

    logger.info("Incremental mapping mode")

    create_empty_db(db_path)
    import_images_with_cameras(image_dir, db_path, camera_mode, image_list, image_options)
    image_ids = get_image_ids(db_path)
    import_features(image_ids, db_path, features_path)
    import_matches(image_ids, db_path, pairs_file, matches_path, min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        estimation_and_geometric_verification(db_path, pairs_file, verbose)

    if pose_prior_csv is not None and pose_prior_confidence != 0:
        set_image_pose_priors(db_path, pose_prior_csv, scale=pose_prior_scale)

    models_dir = sfm_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(models_dir),
        "--Mapper.num_threads", str(min(multiprocessing.cpu_count(), 16))
    ]

    if pose_prior_csv is not None and pose_prior_confidence != 0:
        mapper_cmd += [
            "--Mapper.use_pose_priors", "1",
            "--Mapper.ba_global_weight", str(pose_prior_confidence),
            "--Mapper.ba_global_function", "Cauchy"
        ]

    logger.info("Running COLMAP via subprocess: " + " ".join(mapper_cmd))
    subprocess.run(mapper_cmd, check=True)

    model_folders = sorted(models_dir.glob("*"))

    if not model_folders:
        logger.error("No reconstruction was created by COLMAP.")
        return None

    best_model = model_folders[-1]

    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.move(str(best_model / fname), str(sfm_dir / fname))

    reconstruction = pycolmap.Reconstruction(sfm_dir)
    return reconstruction