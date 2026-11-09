# -*- coding: utf-8 -*-
"""
高精度实时人脸识别 —— insightface SCRFD 检测 + ArcFace 特征

检测: SCRFD (det_500m.onnx) → 5 个精准关键点
特征: w600k_mbf.onnx → 512 维特征
比对: 余弦相似度 + 人脸库

用法:
    python detect_faces_insight.py
    python detect_faces_insight.py --camera 1 --threshold 0.55
操作:
    ESC / q 退出, s 保存截图
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent
WIN_NAME = "Face Recognition (ESC/q quit, s save)"

from face_recognizer import FaceRecognizer


def draw_results(frame, faces, recognizer):
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        det_score = face.det_score

        # 识别
        name, sim = recognizer.recognize(face.embedding)
        if name:
            label = f"{name} {sim:.2f}"
            color = (0, 255, 0)
        else:
            label = f"Stranger {sim:.2f}"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 关键点
        for px, py in face.kps.astype(int):
            cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

    return len(faces)


def open_camera(index, backend="dshow"):
    flag = cv2.CAP_DSHOW if backend == "dshow" else 0
    cap = cv2.VideoCapture(index, flag)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    return None


def main():
    p = argparse.ArgumentParser(description="高精度实时人脸识别 (insightface)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None,
                   help="相似度阈值 (默认取库中配置)")
    p.add_argument("--det-size", type=int, default=480, help="检测输入尺寸 (默认 480, 越小越快)")
    p.add_argument("--skip-frames", type=int, default=2, help="跳帧识别间隔 (默认 2, 隔帧检测+识别)")
    p.add_argument("--db", type=str, default=str(BASE_DIR / "faces_db.json"))
    p.add_argument("--feat-model", type=str,
                   default=str(BASE_DIR / "models" / "buffalo_s" / "w600k_mbf.onnx"))
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=str(BASE_DIR / "shots"))
    args = p.parse_args()

    # insightface 初始化 (SCRFD 检测 + 特征提取)
    print("[INFO] 加载 insightface 模型...")
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    print(f"[INFO] SCRFD + w600k_mbf 加载完成 (检测尺寸 {args.det_size})")

    # 人脸库
    rec = FaceRecognizer(args.feat_model, db_path=args.db,
                         threshold=args.threshold if args.threshold else 0.5)
    if not rec.people:
        print("[WARN] 人脸库为空! 请先运行: python register.py --name 名字")

    # 摄像头
    cap = open_camera(args.camera)
    if cap is None:
        cap = open_camera(args.camera, "msmf")
    if cap is None:
        raise RuntimeError(f"无法打开摄像头 #{args.camera}")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    os.makedirs(args.save_dir, exist_ok=True)

    total_faces = 0
    frame_idx = 0
    fps = 0.0
    t_prev = time.time()
    cached_results = []  # 跳帧缓存: [(bbox, kps, label, color), ...]

    print(f"[INFO] 摄像头 #{args.camera} 已打开 | 跳帧间隔 {args.skip_frames}")
    print("[INFO] 按 ESC / q 退出")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 人脸数变化或到检测帧 --> 完整检测+识别; 否则复用缓存
        if frame_idx % args.skip_frames == 0:
            faces = app.get(frame)
            cached_results = []
            for f in faces:
                name, sim = rec.recognize(f.embedding)
                if name:
                    label = f"{name} {sim:.2f}"
                    color = (0, 255, 0)
                else:
                    label = f"Stranger {sim:.2f}"
                    color = (0, 0, 255)
                cached_results.append((f.bbox.astype(int), f.kps.astype(int), label, color))
        else:
            faces = []  # 复用缓存

        # 绘制 (从缓存)
        for (x1, y1, x2, y2), kps, label, color in cached_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for px, py in kps:
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

        total_faces += len(cached_results)
        frame_idx += 1

        # FPS
        t_now = time.time()
        fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        cv2.putText(frame, f"Faces: {len(cached_results)}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WIN_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        win_open = True
        try:
            win_open = cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            win_open = False
        if key in (ord("q"), ord("Q"), 27) or not win_open:
            print("[INFO] 退出")
            break
        elif key == ord("s"):
            shot = os.path.join(args.save_dir, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(shot, frame)
            print(f"[INFO] 截图已保存: {shot}")
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUMMARY] {frame_idx} 帧, {total_faces} 张人脸, 库中 {len(rec.people)} 人")


if __name__ == "__main__":
    main()
