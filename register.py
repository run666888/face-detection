# -*- coding: utf-8 -*-
"""
人脸注册工具 —— 把一个人登记进人脸库

用法:
    摄像头注册:  python register.py --name 张三
                python register.py --name 张三 --frames 15
    单张图片:    python register.py --name 张三 --image photo.jpg
    批量导入:    python register.py --name 张三 --image-dir ./my_photos
    混合模式:    python register.py --name 张三 --image a.jpg --camera 0

提示: 多种光线/角度注册效果更好
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from face_recognizer import FaceRecognizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEAT_MODEL = BASE_DIR / "models" / "buffalo_s" / "w600k_mbf.onnx"
DEFAULT_DET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="人脸注册")
    p.add_argument("--name", required=True, help="要注册的名字")
    p.add_argument("--camera", type=int, default=-1, help="摄像头编号 (不使用摄像头则不传)")
    p.add_argument("--frames", type=int, default=15, help="摄像头采样帧数 (默认 15)")
    p.add_argument("--image", type=str, default=None, help="从单张图片注册 (需含人脸)")
    p.add_argument("--image-dir", type=str, default=None, help="从图片文件夹批量导入")
    p.add_argument("--feat-model", type=str, default=str(DEFAULT_FEAT_MODEL), help="特征模型路径")
    p.add_argument("--det-model", type=str, default=str(DEFAULT_DET_MODEL), help="检测模型路径")
    p.add_argument("--db", type=str, default=str(BASE_DIR / "faces_db.json"), help="人脸库路径")
    return p.parse_args()


def extract_from_image(img_path, detector, recognizer):
    """从单张图片提取人脸特征, 返回 embedding 或 None"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 无法读取: {img_path}")
        return None
    h, w = img.shape[:2]
    det_img = cv2.resize(img, (640, int(h * 640 / w)))
    detector.setInputSize((det_img.shape[1], det_img.shape[0]))
    _, faces = detector.detect(det_img)
    if faces is None or len(faces) == 0:
        print(f"[WARN] 未检测到人脸: {img_path}")
        return None
    scale = w / det_img.shape[1]
    faces[:, :4] *= scale
    faces[:, 5:] *= scale
    areas = faces[:, 2] * faces[:, 3]
    face = faces[int(np.argmax(areas))]
    x, y, fw, fh = face[:4].astype(int)
    if fw < 80:
        print(f"[WARN] 人脸太小 ({fw}x{fh}): {img_path}")
        return None
    aligned = recognizer.align_face(img, face)
    if aligned is None:
        print(f"[WARN] 对齐失败: {img_path}")
        return None
    return recognizer.get_embedding(aligned)


def main():
    args = parse_args()
    rec = FaceRecognizer(args.feat_model, db_path=args.db)
    detector = cv2.FaceDetectorYN_create(args.det_model, "", (640, 360), 0.6, 0.3, 50)

    embeddings = []

    # ---- 图片注册 ----
    if args.image:
        emb = extract_from_image(args.image, detector, rec)
        if emb is not None:
            embeddings.append(emb)
            print(f"  [OK] {args.image}")

    if args.image_dir:
        dir_path = Path(args.image_dir)
        if not dir_path.is_dir():
            sys.exit(f"目录不存在: {dir_path}")
        files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        print(f"[批量] {dir_path} 中扫描到 {len(files)} 张图片")
        for f in files:
            emb = extract_from_image(str(f), detector, rec)
            if emb is not None:
                embeddings.append(emb)
                print(f"  [OK] {f.name}")

    # ---- 摄像头注册 ----
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            sys.exit(f"无法打开摄像头 #{args.camera}")
        cap.set(cv2.CAP_PROP_FPS, 30)
        cv2.namedWindow("Register - look at the camera", cv2.WINDOW_NORMAL)

        samples = []
        print(f"[注册] 请让 '{args.name}' 正对摄像头, 自动采样 {args.frames} 帧...")
        print("[提示] 按 q 提前退出")

        while len(samples) < args.frames:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 读取失败")
                break
            h, w = frame.shape[:2]
            det_img = cv2.resize(frame, (640, int(h * 640 / w)))
            detector.setInputSize((det_img.shape[1], det_img.shape[0]))
            _, faces = detector.detect(det_img)
            if faces is not None and len(faces) > 0:
                scale = w / det_img.shape[1]
                faces[:, :4] *= scale
                faces[:, 5:] *= scale
                areas = faces[:, 2] * faces[:, 3]
                face = faces[int(np.argmax(areas))]
                x, y, fw, fh = face[:4].astype(int)
                if fw >= 100:
                    aligned = rec.align_face(frame, face)
                    if aligned is not None:
                        samples.append(rec.get_embedding(aligned))
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                        cv2.putText(frame, f"sample {len(samples)}/{args.frames}",
                                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "too small - come closer", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "no face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"registering: {args.name}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Register - look at the camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.05)

        cap.release()
        cv2.destroyAllWindows()
        embeddings.extend(samples)
        if samples:
            print(f"[注册] 摄像头采集 {len(samples)} 帧人脸样本")

    # ---- 入库 ----
    if len(embeddings) == 0:
        sys.exit("[注册] 失败: 没有提取到任何有效人脸特征")

    avg = sum(embeddings) / len(embeddings)
    avg /= np.linalg.norm(avg)
    rec.add_person(args.name, avg)
    print(f"[注册] 完成: '{args.name}' 已加入人脸库 (基于 {len(embeddings)} 个样本)")


if __name__ == "__main__":
    main()
