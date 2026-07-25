from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from transformers import AutoTokenizer

DIMENSION = 1024
IMAGE_SIZE = 512
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
QUERY_PREFIX = "Represent the query for retrieving evidence documents: "
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def emit(value: dict) -> None:
    # Prefix and Base64 framing keep native ONNX/CUDA log output, Windows paths,
    # control characters and console encodings from corrupting the JSON protocol.
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    framed = "@@JSON@@" + base64.b64encode(payload).decode("ascii")
    print(framed, flush=True)


def progress(current: int, total: int, message: str) -> None:
    emit({"event": "progress", "current": current, "total": total, "message": message})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddingEngine:
    def __init__(self, app_dir: Path) -> None:
        self.app_dir = app_dir.resolve()
        self.data_dir = self._find_data_dir()
        self.model_dir = self._find_model_dir()
        self.db_path = self.data_dir / "image_embedding.db"
        self.index_path = self.data_dir / "image_embedding.faiss"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, use_fast=True,
            fix_mistral_regex=True,
        )
        self.session, self.device = self._create_session()
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        if not {"input_ids", "pixel_values"}.issubset(self.input_names):
            raise RuntimeError(f"未対応のONNX入力です: {sorted(self.input_names)}")
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIMENSION))
        self._rebuild_index()

    def _find_data_dir(self) -> Path:
        for candidate in [self.app_dir, *self.app_dir.parents]:
            if (candidate / "ImageEmbedding.csproj").is_file():
                return candidate
        return self.app_dir

    def _find_model_dir(self) -> Path:
        candidates = [
            self.app_dir / "jinaai_jina-clip-v2",
            self.app_dir.parent / "jinaai_jina-clip-v2",
        ]
        candidates.extend(parent / "jinaai_jina-clip-v2" for parent in self.app_dir.parents)
        for candidate in candidates:
            if (candidate / "onnx" / "model_fp16.onnx").is_file() and (candidate / "tokenizer.json").is_file():
                return candidate
        raise FileNotFoundError("jinaai_jina-clip-v2 フォルダ（model_fp16.onnx と tokenizer.json）を見つけられません。")

    def _create_session(self) -> tuple[ort.InferenceSession, str]:
        model_path = str(self.model_dir / "onnx" / "model_fp16.onnx")
        available = ort.get_available_providers()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        force_cpu = os.environ.get("JINA_CLIP_FORCE_CPU", "").lower() in {"1", "true", "yes"}
        if "CUDAExecutionProvider" in available and not force_cpu:
            try:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=options,
                    providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                )
                if "CUDAExecutionProvider" in session.get_providers():
                    return session, "CUDA (GPU)"
            except Exception as error:
                emit({"event": "log", "message": f"CUDAを初期化できないためCPUへ切り替えます: {str(error).splitlines()[0]}"})
        # The optimized FP16 graph triggers a SimplifiedLayerNormFusion issue on the CPU EP.
        # Disabling graph rewrites keeps the official model portable to machines without CUDA.
        cpu_options = ort.SessionOptions()
        cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        cpu_options.log_severity_level = 3
        return ort.InferenceSession(model_path, sess_options=cpu_options, providers=["CPUExecutionProvider"]), "CPU"

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE COLLATE NOCASE,
                file_size INTEGER NOT NULL,
                modified_utc INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_images_path ON images(path);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR REPLACE INTO metadata(key, value) VALUES('model', 'jinaai/jina-clip-v2');
            INSERT OR REPLACE INTO metadata(key, value) VALUES('dimension', '1024');
            INSERT OR REPLACE INTO metadata(key, value) VALUES('metric', 'cosine (normalized inner product)');
            """
        )
        self.connection.commit()

    def _rebuild_index(self) -> None:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIMENSION))
        rows = self.connection.execute("SELECT id, embedding FROM images ORDER BY id").fetchall()
        if rows:
            vectors = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
            ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
            if vectors.shape[1] != DIMENSION:
                raise RuntimeError("SQLite内のEmbedding次元がモデルと一致しません。")
            index.add_with_ids(np.ascontiguousarray(vectors), ids)
        self.index = index
        faiss.write_index(self.index, str(self.index_path))

    @staticmethod
    def _discover(paths: Iterable[str]) -> list[Path]:
        found: dict[str, Path] = {}
        for raw in paths:
            path = Path(raw).resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                found[str(path).lower()] = path
            elif path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                        found[str(child.resolve()).lower()] = child.resolve()
        return sorted(found.values(), key=lambda item: str(item).lower())

    @staticmethod
    def _prepare_image(path: Path) -> tuple[np.ndarray, int, int]:
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            width, height = source.size
            scale = IMAGE_SIZE / min(width, height)
            resized = source.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
            left = max(0, (resized.width - IMAGE_SIZE) // 2)
            top = max(0, (resized.height - IMAGE_SIZE) // 2)
            cropped = resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
            pixels = np.asarray(cropped, dtype=np.float32) / 255.0
            pixels = (pixels - MEAN) / STD
            pixels = np.transpose(pixels, (2, 0, 1))[None, ...]
            return np.ascontiguousarray(pixels), width, height

    def _dummy_text(self) -> np.ndarray:
        return np.asarray([[0, 2]], dtype=np.int64)

    @staticmethod
    def _dummy_image() -> np.ndarray:
        return np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    def _run(self, input_ids: np.ndarray, pixel_values: np.ndarray) -> list[np.ndarray]:
        feed = {"input_ids": input_ids, "pixel_values": pixel_values}
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.ones_like(input_ids, dtype=np.int64)
        return self.session.run(None, feed)

    def _find_embedding(self, outputs: list[np.ndarray], kind: str) -> np.ndarray:
        preferred = [
            index for index, name in enumerate(self.output_names)
            if "l2norm" in name.lower() and kind in name.lower()
        ]
        if not preferred:
            candidates = [index for index, value in enumerate(outputs) if value.ndim == 2 and value.shape[-1] == DIMENSION]
            if len(candidates) >= 4:
                preferred = [candidates[2 if kind == "text" else 3]]
            elif len(candidates) >= 2:
                preferred = [candidates[-2 if kind == "text" else -1]]
        if not preferred:
            raise RuntimeError(f"{kind} Embedding出力をONNXモデルから取得できません。出力: {self.output_names}")
        vector = np.asarray(outputs[preferred[0]][0], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise RuntimeError("モデルがゼロベクトルを返しました。")
        return np.ascontiguousarray(vector / norm)

    def embed_image(self, path: Path) -> tuple[np.ndarray, int, int]:
        pixels, width, height = self._prepare_image(path)
        outputs = self._run(self._dummy_text(), pixels)
        return self._find_embedding(outputs, "image"), width, height

    def embed_text(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            QUERY_PREFIX + text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=8192,
        )
        ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        outputs = self._run(ids, self._dummy_image())
        return self._find_embedding(outputs, "text")

    def index_paths(self, paths: list[str]) -> dict:
        images = self._discover(paths)
        total = len(images)
        added = updated = skipped = failed = 0
        errors: list[str] = []
        for current, path in enumerate(images, start=1):
            progress(current - 1, total, f"Embedding中 ({current}/{total}): {path.name}")
            try:
                stat = path.stat()
                row = self.connection.execute(
                    "SELECT id, file_size, modified_utc FROM images WHERE path = ?", (str(path),)
                ).fetchone()
                modified = int(stat.st_mtime_ns)
                if row and row["file_size"] == stat.st_size and row["modified_utc"] == modified:
                    skipped += 1
                    progress(current, total, f"変更なし ({current}/{total}): {path.name}")
                    continue
                vector, width, height = self.embed_image(path)
                now = utc_now()
                if row:
                    self.connection.execute(
                        "UPDATE images SET file_size=?, modified_utc=?, width=?, height=?, embedding=?, updated_at=? WHERE id=?",
                        (stat.st_size, modified, width, height, vector.tobytes(), now, row["id"]),
                    )
                    updated += 1
                else:
                    self.connection.execute(
                        "INSERT INTO images(path,file_size,modified_utc,width,height,embedding,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                        (str(path), stat.st_size, modified, width, height, vector.tobytes(), now, now),
                    )
                    added += 1
                self.connection.commit()
                progress(current, total, f"登録済み ({current}/{total}): {path.name}")
            except Exception as error:
                failed += 1
                errors.append(f"{path}: {error}")
                progress(current, total, f"処理失敗 ({current}/{total}): {path.name}")
        self._rebuild_index()
        return {"added": added, "updated": updated, "skipped": skipped, "failed": failed,
                "errors": errors[:20], "count": self.count()}

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def _search(self, vector: np.ndarray, limit: int) -> dict:
        if self.index.ntotal == 0:
            return {"items": []}
        limit = min(max(1, limit), int(self.index.ntotal))
        scores, ids = self.index.search(vector.reshape(1, -1), limit)
        rows_by_id = {
            row["id"]: row for row in self.connection.execute(
                f"SELECT id,path,width,height FROM images WHERE id IN ({','.join('?' for _ in ids[0])})",
                [int(item) for item in ids[0]],
            ).fetchall()
        }
        items = []
        for item_id, score in zip(ids[0], scores[0]):
            row = rows_by_id.get(int(item_id))
            if row:
                items.append({"id": row["id"], "path": row["path"], "width": row["width"],
                              "height": row["height"], "score": float(score)})
        return {"items": items}

    def search_text(self, query: str, limit: int) -> dict:
        return self._search(self.embed_text(query), limit)

    def search_image(self, path: str, limit: int) -> dict:
        vector, _, _ = self.embed_image(Path(path).resolve())
        return self._search(vector, limit)

    def list_images(self, limit: int) -> dict:
        rows = self.connection.execute(
            "SELECT id,path,width,height FROM images ORDER BY updated_at DESC LIMIT ?", (min(max(1, limit), 10000),)
        ).fetchall()
        return {"items": [{"id": row["id"], "path": row["path"], "width": row["width"],
                            "height": row["height"], "score": -1.0} for row in rows]}

    def delete(self, ids: list[int]) -> dict:
        if ids:
            placeholders = ",".join("?" for _ in ids)
            self.connection.execute(f"DELETE FROM images WHERE id IN ({placeholders})", [int(item) for item in ids])
            self.connection.commit()
            self._rebuild_index()
        return {"count": self.count()}

    def close(self) -> None:
        self.connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", required=True)
    args = parser.parse_args()
    engine: EmbeddingEngine | None = None
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            request_id = request["id"]
            command = request["command"]
            payload = request.get("payload", {})
            if command == "init":
                engine = EmbeddingEngine(Path(args.app_dir))
                result = {"ready": True, "device": engine.device}
            elif engine is None:
                raise RuntimeError("ワーカーが初期化されていません。")
            elif command == "status":
                result = {"device": engine.device, "count": engine.count(), "database": str(engine.db_path)}
            elif command == "index":
                result = engine.index_paths(payload["paths"])
            elif command == "search_text":
                result = engine.search_text(payload["query"], int(payload.get("limit", 200)))
            elif command == "search_image":
                result = engine.search_image(payload["path"], int(payload.get("limit", 200)))
            elif command == "list":
                result = engine.list_images(int(payload.get("limit", 1000)))
            elif command == "delete":
                result = engine.delete(payload["ids"])
            elif command == "shutdown":
                engine.close()
                emit({"id": request_id, "ok": True, "result": {}})
                return 0
            else:
                raise ValueError(f"不明なコマンドです: {command}")
            emit({"id": request_id, "ok": True, "result": result})
        except Exception as error:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            emit({"id": request.get("id", ""), "ok": False, "error": str(error)})
    if engine is not None:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
