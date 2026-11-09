# -*- coding: utf-8 -*-
"""
人脸识别核心模块
- 特征提取: MobileFaceNet (w600k_mbf.onnx, insightface 标准模型, 512 维特征)
- 人脸对齐: 5 点仿射变换 (YuNet 关键点 -> ArcFace 标准模板)
- 比对: 余弦相似度
- 人脸库: faces_db.json (名字 -> embedding 列表)
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "faces_db.json"

# ArcFace 标准 5 点对齐模板 (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻尖
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041],  # 右嘴角
], dtype=np.float32)

# YuNet 关键点顺序: 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角 (与模板一致)
INPUT_SIZE = 112
FEATURE_DIM = 512
DEFAULT_THRESHOLD = 0.5  # 余弦相似度阈值, 高于视为同一人


class FaceRecognizer:
    """封装特征模型与人脸库"""

    def __init__(self, model_path, db_path=None, threshold=DEFAULT_THRESHOLD):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"特征模型不存在: {model_path}")
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.threshold = threshold
        self.people = {}  # 名字 -> {"embeddings": [512d...], "registered_at": str}
        self._load_db()
        print(f"[识别] 特征模型: {os.path.basename(model_path)} | 库中人员: {list(self.people)}")

    # ---------- 人脸库 ----------
    def _load_db(self):
        if self.db_path.exists():
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            self.people = data.get("people", {})
            for p in self.people.values():
                p["embeddings"] = [np.array(e, dtype=np.float32) for e in p["embeddings"]]

    def save_db(self):
        data = {
            "version": 1,
            "threshold": self.threshold,
            "people": {
                name: {
                    "embeddings": [e.tolist() for e in p["embeddings"]],
                    "registered_at": p["registered_at"],
                }
                for name, p in self.people.items()
            },
        }
        self.db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[识别] 人脸库已保存: {self.db_path}")

    def add_person(self, name, embedding):
        """注册/追加一个人的特征样本"""
        if name not in self.people:
            self.people[name] = {"embeddings": [], "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.people[name]["embeddings"].append(
            embedding.astype(np.float32).reshape(-1))
        self.save_db()
        print(f"[识别] 已登记 '{name}' (累计 {len(self.people[name]['embeddings'])} 个样本)")

    # ---------- 特征提取 ----------
    @staticmethod
    def align_face(frame, face):
        """用 5 个关键点把人脸仿射对齐到 112x112, 返回 RGB 图或 None"""
        pts = face[5:].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(pts, ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(frame, M, (INPUT_SIZE, INPUT_SIZE),
                                 borderMode=cv2.BORDER_REPLICATE)
        return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    def get_embedding(self, face_img_rgb):
        """输入 112x112 RGB 对齐人脸, 输出 L2 归一化 512 维特征"""
        img = (face_img_rgb.astype(np.float32) - 127.5) / 128.0
        blob = img.transpose(2, 0, 1)[None]  # 1x3x112x112
        out = self.session.run(None, {self.input_name: blob})[0]
        emb = out.reshape(-1).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb

    # ---------- 比对 ----------
    def recognize(self, embedding):
        """与库中所有人比对, 返回 (名字, 最高相似度) 或 (None, 最高相似度)"""
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_name, best_score = None, -1.0
        for name, p in self.people.items():
            for ref in p["embeddings"]:
                score = float(np.dot(embedding, ref))  # 均已 L2 归一化 => 余弦
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < self.threshold:
            return None, best_score
        return best_name, best_score
