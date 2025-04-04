import sqlite3
import numpy as np
from pathlib import Path

MAX_IMAGE_ID = 2 ** 31 - 1

TABLES = {
    "cameras": """
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        )""",
    "images": f"""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
            prior_tx REAL, prior_ty REAL, prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 AND image_id < {MAX_IMAGE_ID}),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )""",
    "keypoints": """
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )""",
    "descriptors": """
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        )""",
    "matches": """
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB
        )""",
    "two_view_geometries": """
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB, E BLOB, H BLOB,
            qvec BLOB, tvec BLOB
        )""",
    "name_index": """
        CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)
    """
}

def image_ids_to_pair_id(id1, id2):
    return min(id1, id2) * MAX_IMAGE_ID + max(id1, id2)

def pair_id_to_image_ids(pair_id):
    id2 = pair_id % MAX_IMAGE_ID
    id1 = (pair_id - id2) // MAX_IMAGE_ID
    return id1, id2

def array_to_blob(array: np.ndarray) -> bytes:
    return array.astype(array.dtype, copy=False).tobytes()

def blob_to_array(blob: bytes, dtype, shape=(-1,)) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path: Path):
        return sqlite3.connect(str(database_path), factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables()

    def create_tables(self):
        for name, sql in TABLES.items():
            self.executescript(sql)

    def add_camera(self, model, width, height, params, prior_focal_length=False, camera_id=None):
        params_blob = array_to_blob(np.asarray(params, dtype=np.float64))
        values = (camera_id, model, width, height, params_blob, int(prior_focal_length))
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)", values
        )
        return cursor.lastrowid

    def add_image(self, name, camera_id, prior_q=None, prior_t=None, image_id=None):
        if prior_q is None:
            prior_q = [np.nan] * 4
        if prior_t is None:
            prior_t = [np.nan] * 3
        values = (image_id, name, camera_id, *prior_q, *prior_t)
        cursor = self.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints: np.ndarray):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        rows, cols = keypoints.shape
        data = array_to_blob(keypoints)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, rows, cols, data)
        )

    def add_descriptors(self, image_id, descriptors: np.ndarray):
        descriptors = np.asarray(descriptors, dtype=np.uint8)
        rows, cols = descriptors.shape
        data = array_to_blob(descriptors)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id, rows, cols, data)
        )

    def add_matches(self, image_id1, image_id2, matches: np.ndarray):
        matches = np.asarray(matches, dtype=np.uint32)
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        rows, cols = matches.shape
        data = array_to_blob(matches)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id, rows, cols, data)
        )

    def add_two_view_geometry(
        self, image_id1, image_id2, matches,
        F=np.eye(3), E=np.eye(3), H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3), config=2
    ):
        matches = np.asarray(matches, dtype=np.uint32)
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        rows, cols = matches.shape
        data_blob = array_to_blob(matches)
        F_blob = array_to_blob(np.asarray(F, dtype=np.float64))
        E_blob = array_to_blob(np.asarray(E, dtype=np.float64))
        H_blob = array_to_blob(np.asarray(H, dtype=np.float64))
        qvec_blob = array_to_blob(np.asarray(qvec, dtype=np.float64))
        tvec_blob = array_to_blob(np.asarray(tvec, dtype=np.float64))
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, rows, cols, data_blob, config, F_blob, E_blob, H_blob, qvec_blob, tvec_blob)
        )
