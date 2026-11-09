# -*- coding: utf-8 -*-
"""
多目标实时人脸检测 —— 调用摄像头实时检测画面中的多张人脸

模型: OpenCV Zoo YuNet (face_detection_yunet_2023mar.onnx)
  - 支持多张人脸同时检测
  - 输出: 人脸框 + 置信度 + 5 个关键点(双眼/鼻尖/左右嘴角)

用法:
    python detect_faces.py                       # 默认摄像头 0
    python detect_faces.py --camera 1            # 指定摄像头
    python detect_faces.py --score 0.5           # 调低置信度阈值(更易检出)
    python detect_faces.py --no-display --max-frames 30   # 无窗口测试模式

操作:
    q  退出
    s  保存当前帧截图到 shots/ 目录
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from face_recognizer import FaceRecognizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_FEAT_MODEL = BASE_DIR / "models" / "buffalo_s" / "w600k_mbf.onnx"
WIN_NAME = "Face Detect (q/ESC quit, s save)"


def parse_args():
    p = argparse.ArgumentParser(description="多目标实时人脸检测 (YuNet)")
    p.add_argument("--camera", type=int, default=0, help="摄像头编号 (默认 0)")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "msmf", "dshow"],
                   help="摄像头后端: auto 自动回退 (默认), msmf, dshow")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="YuNet ONNX 模型路径")
    p.add_argument("--score", type=float, default=0.6, help="置信度阈值 (默认 0.6)")
    p.add_argument("--nms", type=float, default=0.3, help="NMS 阈值 (默认 0.3)")
    p.add_argument("--topk", type=int, default=50, help="最多检测人脸数 (默认 50)")
    p.add_argument("--recognize", action="store_true", help="启用识别模式(认出是谁)")
    p.add_argument("--feat-model", type=str, default=str(DEFAULT_FEAT_MODEL), help="特征提取模型")
    p.add_argument("--db", type=str, default=str(BASE_DIR / "faces_db.json"), help="人脸库路径")
    p.add_argument("--threshold", type=float, default=None, help="识别相似度阈值 (默认取人脸库配置)")
    p.add_argument("--recognize-every", type=int, default=3, help="识别跳帧数, 每 N 帧识别一次 (默认 3, 提升帧率)")
    p.add_argument("--width", type=int, default=640, help="检测输入宽度 (默认 640)")
    p.add_argument("--max-frames", type=int, default=0, help="处理帧数上限, 0=无限 (测试用)")
    p.add_argument("--no-display", action="store_true", help="不显示窗口, 仅打印检测结果")
    p.add_argument("--save-dir", type=str, default=str(BASE_DIR / "shots"), help="截图保存目录")
    return p.parse_args()


def recognize_faces(frame, faces, recognizer):
    """对每张人脸提取特征并比对, 返回 [(name, sim), ...]"""
    results = []
    for face in faces:
        aligned = recognizer.align_face(frame, face)
        if aligned is not None:
            name, sim = recognizer.recognize(recognizer.get_embedding(aligned))
        else:
            name, sim = None, 0.0
        results.append((name, sim))
    return results


def draw_faces(frame, faces, score_thr, labels=None):
    """在帧上绘制人脸框、置信度与关键点; labels 为每张人脸的 (名字, 相似度) 缓存"""
    for i, face in enumerate(faces):
        x, y, w, h = face[:4].astype(int)
        score = float(face[4])
        if score < score_thr:
            continue

        # 识别标签 (来自缓存): 认识的人绿框+名字, 陌生人红框
        label = f"{score:.2f}"
        color = (0, 255, 0)
        if labels is not None and i < len(labels):
            name, sim = labels[i]
            if name:
                label = f"{name} {sim:.2f}"
            else:
                label = f"Stranger {sim:.2f}"
                color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # 标签背景 + 文字
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # 5 个关键点: 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角
        for j, pt in enumerate(face[5:].reshape(5, 2)):
            px, py = pt.astype(int)
            cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
            cv2.putText(frame, str(j + 1), (px + 3, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return len(faces)


def open_camera(index, backend):
    """打开摄像头: auto 模式在 MSMF 失败/黑帧时回退 DSHOW"""
    attempts = []
    if backend in ("auto", "dshow"):
        attempts.append(("DSHOW", cv2.CAP_DSHOW))
    if backend in ("auto", "msmf"):
        attempts.append(("MSMF", 0))
    for name, flag in attempts:
        cap = cv2.VideoCapture(index, flag) if flag else cv2.VideoCapture(index)
        if cap.isOpened():
            ok = False
            for _ in range(3):
                try:
                    ret, frame = cap.read()
                except cv2.error as e:
                    print(f"[WARN] 后端 {name} 读帧异常: {e}")
                    ret, frame = False, None
                if ret and frame is not None and frame.size > 0:
                    ok = True
                    break
            if ok:
                # 强制帧率, 避免暗光自动曝光导致 1 FPS
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"[INFO] 摄像头后端: {name}")
                return cap
            print(f"[WARN] 后端 {name} 读不到帧, 尝试下一个")
        cap.release()
    return None


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}\n请先下载 YuNet 模型放入 models/ 目录")

    # 打开摄像头
    cap = open_camera(args.camera, args.backend)
    if cap is None:
        raise RuntimeError(f"无法打开摄像头 #{args.camera}, 请检查设备、占用进程或换 --camera/--backend")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 初始化 YuNet 检测器
    detector = cv2.FaceDetectorYN_create(
        args.model, "", (args.width, args.width * 9 // 16),
        args.score, args.nms, args.topk,
    )

    # 识别模式: 加载特征模型与人脸库
    recognizer = None
    if args.recognize:
        recognizer = FaceRecognizer(
            args.feat_model, db_path=args.db,
            threshold=args.threshold if args.threshold is not None else 0.5)
        if not recognizer.people:
            print("[WARN] 人脸库为空! 请先运行: python register.py --name 你的名字")

    if not args.no_display:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    os.makedirs(args.save_dir, exist_ok=True)
    total_faces = 0
    frame_idx = 0
    fps = 0.0
    t_prev = time.time()
    rec_cache = []  # 识别结果缓存: [(name, sim), ...]
    dark_warn = 0   # 暗帧连续计数

    print(f"[INFO] 摄像头 #{args.camera} 已打开 | 模型: {os.path.basename(args.model)}")
    print("[INFO] 按 q / ESC 退出 (或直接关闭窗口), 按 s 保存截图")
    if recognizer:
        print(f"[INFO] 识别模式: 每 {args.recognize_every} 帧识别一次 (人脸数变化时立即识别)")
    print("-" * 50)

    while True:
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            print(f"[WARN] 读取摄像头帧异常: {e}")
            print("[WARN] 摄像头驱动状态异常, 建议: 重启电脑 或 检查摄像头是否被其他程序占用")
            break
        if not ret or frame is None:
            print("[WARN] 读取摄像头帧失败, 退出")
            break

        # 画面过暗检测 (每 15 帧检查一次, 连续过暗只提示一次)
        if frame_idx < 180 and frame_idx % 15 == 0 and np.mean(frame) < 8:
            dark_warn += 1
            if dark_warn >= 3:
                print(f"[WARN] 画面持续过暗! 当前亮度 {np.mean(frame):.1f}/255")
                print("       请检查: 1) 按 F9/Fn+F9 打开摄像头开关")
                print("               2) 摄像头镜头是否被遮挡/贴纸/污渍")
                print("               3) 环境光线是否充足")
                dark_warn = -9999

        # 保持宽高比缩放到检测宽度
        h, w = frame.shape[:2]
        scale = args.width / w
        det_w, det_h = args.width, int(h * scale)
        det_img = cv2.resize(frame, (det_w, det_h))

        detector.setInputSize((det_w, det_h))
        _, faces = detector.detect(det_img)  # faces: Nx15 (bbox4 + score + 5*2 关键点)

        labels = None
        if recognizer is not None:
            n_now = 0 if faces is None else len(faces)
            # 人脸数变化或到识别帧时重新识别, 否则复用缓存 (跳帧提速)
            if n_now != len(rec_cache) or frame_idx % args.recognize_every == 0:
                rec_cache = recognize_faces(frame, faces, recognizer) if n_now else []
            labels = rec_cache if n_now == len(rec_cache) else None

        n = 0
        if faces is not None:
            # 关键点坐标要按缩放比例映射回原始帧
            faces[:, 5:] /= scale
            faces[:, :4] /= scale
            n = draw_faces(frame, faces, args.score, labels)

        total_faces += n
        frame_idx += 1

        # FPS 统计
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        if args.no_display:
            if frame_idx % 10 == 0:
                print(f"[帧 {frame_idx}] 检测到 {n} 张人脸 | FPS: {fps:.1f}")
            if args.max_frames and frame_idx >= args.max_frames:
                break
            continue

        # 显示信息
        mode = "RECOGNIZE" if recognizer else "DETECT"
        cv2.putText(frame, f"[{mode}] Faces: {n}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WIN_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        # 窗口是否仍打开 (用户点 X 关闭时也退出)
        win_open = True
        try:
            win_open = cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            win_open = False
        if key in (ord("q"), ord("Q"), 27) or not win_open:
            print("[INFO] 用户退出")
            break
        elif key == ord("s"):
            shot = os.path.join(args.save_dir, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(shot, frame)
            print(f"[INFO] 截图已保存: {shot}")
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"[INFO] 已达帧数上限 {args.max_frames}, 自动退出")
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"[SUMMARY] 共处理 {frame_idx} 帧, 累计检测到 {total_faces} 张人脸")
    if recognizer:
        print(f"[SUMMARY] 识别模式: 人脸库 {len(recognizer.people)} 人 | 相似度阈值 {recognizer.threshold:.2f}")


if __name__ == "__main__":
    main()
