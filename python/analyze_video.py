"""
AI 微表情測謊分析 — Python 參考實作 (OpenCV + MediaPipe + DeepFace)

Same pipeline as the browser demo (app.js), written with the libraries named in the
project proposal. Reads a video, writes an annotated copy plus a per-frame CSV.

    pip install -r requirements.txt
    python analyze_video.py input.mp4 -o output.mp4

Download the landmark model first (once):
    curl -LO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

NOTE: reference code for the report; the class demo uses the browser version.
"""

import argparse
import csv
import math

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

CAL_SECONDS = 10
EMOTION_EVERY_N_FRAMES = 6  # DeepFace is slow; run it a few times per second

# weight in score, std floor (same values as app.js)
CUES = {
    "brow":  (1.0, 0.04),
    "gaze":  (1.0, 0.05),
    "blink": (0.6, None),
    "lip":   (0.7, 0.04),
    "head":  (0.5, 4.0),
    "fear":  (0.8, 0.06),
    "shift": (0.5, 0.06),
}
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def ema(prev, x, dt, tau):
    return x if prev is None else prev + (x - prev) * (1 - math.exp(-dt / tau))


def blendshape_dict(result):
    if not result.face_blendshapes:
        return {}
    return {c.category_name: c.score for c in result.face_blendshapes[0]}


def head_angles(result):
    if not result.facial_transformation_matrixes:
        return None
    m = np.array(result.facial_transformation_matrixes[0])
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -m[2, 0]))))
    pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
    return yaw, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--output", default="annotated.mp4")
    ap.add_argument("--model", default="face_landmarker.task")
    args = ap.parse_args()

    # 1. OpenCV: read frames
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 2. MediaPipe Face Landmarker: 478 landmarks + 52 blendshapes (≈ FACS action units)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    dt = 1 / fps
    sm = {}
    samples = {k: [] for k in ["brow", "lip", "head", "fear", "shift", "gh", "gv"]}
    base = None
    zs = {k: 0.0 for k in CUES}
    score = 0.0
    blinks, blink_closed = [], False
    head_prev, head_speed = None, 0.0
    emo_prev, fear, shift = None, 0.0, 0.0
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
        b = blendshape_dict(result)

        if b:
            g = lambda k: b.get(k, 0.0)
            brow = max(g("browInnerUp"), (g("browOuterUpLeft") + g("browOuterUpRight")) / 2)
            blink_v = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2
            if not blink_closed and blink_v > 0.5:
                blink_closed = True
                blinks.append(t)
            elif blink_closed and blink_v < 0.3:
                blink_closed = False
            blinks = [x for x in blinks if t - x < 15]
            gh = (g("eyeLookOutLeft") + g("eyeLookInRight") - g("eyeLookInLeft") - g("eyeLookOutRight")) / 2
            gv = (g("eyeLookUpLeft") + g("eyeLookUpRight") - g("eyeLookDownLeft") - g("eyeLookDownRight")) / 2
            lip = max((g("mouthPressLeft") + g("mouthPressRight")) / 2, (g("mouthRollLower") + g("mouthRollUpper")) / 2)
            ang = head_angles(result)
            if ang and head_prev:
                head_speed = ema(head_speed, math.hypot(ang[0] - head_prev[0], ang[1] - head_prev[1]) / dt, dt, 0.4)
            head_prev = ang or head_prev

            # 3. DeepFace: 7-class emotion
            if frame_idx % EMOTION_EVERY_N_FRAMES == 0:
                try:
                    r = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)[0]
                    e = np.array([r["emotion"][k] / 100 for k in EMOTIONS])
                    if emo_prev is not None:
                        shift = float(np.abs(e - emo_prev).sum() / 2)
                    emo_prev = e
                    fear = r["emotion"]["fear"] / 100 + 0.7 * r["emotion"]["surprise"] / 100
                except Exception:
                    pass

            for k, v in [("brow", brow), ("lip", lip), ("fear", fear), ("shift", shift)]:
                sm[k] = ema(sm.get(k), v, dt, 0.25)
            sm["head"] = head_speed
            sm["gh"] = ema(sm.get("gh"), gh, dt, 0.15)
            sm["gv"] = ema(sm.get("gv"), gv, dt, 0.15)

            # 4. Baseline (first CAL_SECONDS) — like the polygraph's control questions
            if base is None:
                for k in samples:
                    samples[k].append(sm[k])
                if t >= CAL_SECONDS and len(samples["brow"]) >= 30:
                    base = {k: (float(np.mean(samples[k])), float(np.std(samples[k]))) for k in samples}
                    gd = np.hypot(np.array(samples["gh"]) - base["gh"][0], np.array(samples["gv"]) - base["gv"][0])
                    base["gaze"] = (float(gd.mean()), float(gd.std()))
                    base["blink_rate"] = len(blinks) / t * 60
            else:
                # 5. z-scores → weighted score → 0–100
                z = {}
                for k in ["brow", "lip", "head", "fear", "shift"]:
                    mean, std = base[k]
                    z[k] = (sm[k] - mean) / max(std, CUES[k][1])
                gd = math.hypot(sm["gh"] - base["gh"][0], sm["gv"] - base["gv"][0])
                z["gaze"] = (gd - base["gaze"][0]) / max(base["gaze"][1], CUES["gaze"][1])
                rate = len(blinks) / 15 * 60
                z["blink"] = (rate - base["blink_rate"]) / max(8, base["blink_rate"] * 0.6)
                s = 0.0
                for k, (wgt, _) in CUES.items():
                    zs[k] = ema(zs[k], min(5.0, max(0.0, z[k])), dt, 0.5)
                    s += wgt * max(0.0, zs[k] - 1)
                score = ema(score, 100 / (1 + math.exp(-(s - 1.8) * 1.3)), dt, 0.6)

        # overlay
        label = f"Likelihood index: {score:4.0f}%" if base else f"Calibrating baseline... {max(0, CAL_SECONDS - t):.0f}s"
        color = (214, 242, 92) if score < 40 else (71, 181, 255) if score < 70 else (109, 77, 255)
        cv2.rectangle(frame, (16, 16), (430, 70), (15, 10, 5), -1)
        cv2.putText(frame, label, (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y = 100
        for k in CUES:
            if zs[k] > 1.8:
                cv2.putText(frame, f"^ {k}", (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (109, 77, 255), 2, cv2.LINE_AA)
                y += 30
        out.write(frame)
        rows.append({"t": round(t, 3), "score": round(score, 1), **{f"z_{k}": round(v, 2) for k, v in zs.items()}})
        frame_idx += 1

    cap.release()
    out.release()
    with open(args.output.rsplit(".", 1)[0] + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["t"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {args.output} and CSV ({frame_idx} frames)")


if __name__ == "__main__":
    main()
