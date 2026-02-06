import cv2
import mediapipe as mp
import time
import os
import urllib.request
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 478点ランドマークの中から必要な点だけ使う
# 口角: left=61, right=291
# 眉は候補点から「眉頭（内側）」を動的に選ぶ
KEYPOINTS = {
    "mouth_left": 61,
    "mouth_right": 291,
}

# 眉の候補点（左/右の眉ライン上）
# ここから「顔の中心に近い点」を眉頭として選ぶ
BROW_LEFT_CANDIDATES = [70, 63, 105, 66, 107]
BROW_RIGHT_CANDIDATES = [300, 293, 334, 296, 336]

# 鼻の下（基準点）として使うランドマーク
# 必要なら別番号に変更してください
NOSE_BASE_INDEX = 2

# 低負荷で動かすためのキャプチャ設定
FRAME_W, FRAME_H = 640, 480
PRINT_INTERVAL_SEC = 0.2  # コンソール出力の間隔（秒）
DEFORM_POINTS_STEP = 1  # 変形用の点数を間引くなら2以上
DEFORM_STRENGTH = 2.0  # 変形強度(1.0〜3.0)

# Face LandmarkerモデルのURLと保存先
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join("models", "face_landmarker.task")


# 役割: モデルファイルの存在確認と自動ダウンロード
def ensure_model(path: str, url: str) -> None:
    """モデルが未存在ならダウンロードする。"""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading model...")
    urllib.request.urlretrieve(url, path)


# 役割: ランドマークの正規化座標をピクセル配列に変換
def landmarks_to_points(landmarks, w, h):
    """正規化座標(0-1)をピクセル座標に変換して配列化する。"""
    return np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)


# 役割: Delaunay三角形分割のインデックスを生成
def build_delaunay_triangles(rect, points):
    """Delaunay三角形分割のインデックス配列を作る。"""
    subdiv = cv2.Subdiv2D(rect)
    x0, y0, w, h = rect
    x1, y1 = x0 + w - 1, y0 + h - 1
    safe_points = []
    for p in points:
        # Subdiv2Dは範囲外の点で落ちるためクランプ
        px = min(max(float(p[0]), x0), x1)
        py = min(max(float(p[1]), y0), y1)
        safe_points.append((px, py))
        subdiv.insert((px, py))

    tri_list = subdiv.getTriangleList()
    tri_indices = []

    point_map = {}
    for i, p in enumerate(safe_points):
        key = (int(round(p[0])), int(round(p[1])))
        point_map[key] = i

    for t in tri_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for (x, y) in pts:
            key = (int(round(x)), int(round(y)))
            if key in point_map:
                idx.append(point_map[key])
            else:
                # 近傍点を探す（完全一致しない場合のフォールバック）
                dmin, imin = 1e9, -1
                for i, p in enumerate(safe_points):
                    d = (p[0] - x) ** 2 + (p[1] - y) ** 2
                    if d < dmin:
                        dmin, imin = d, i
                if imin >= 0:
                    idx.append(imin)
        if len(idx) == 3:
            tri_indices.append(tuple(idx))

    return tri_indices


# 役割: 三角形単位でアフィン変換を行い画像に合成
def warp_triangle(img, out, t_src, t_dst):
    """1つの三角形をアフィン変換して合成する。"""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    # サイズが不正な場合はスキップ
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if img1_rect.size == 0:
        return
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(
        img1_rect,
        warp_mat,
        (r2[2], r2[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    out_region = out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    # サイズ不一致の保険（まれに境界で1pxずれる）
    h = min(out_region.shape[0], img2_rect.shape[0], mask.shape[0])
    w = min(out_region.shape[1], img2_rect.shape[1], mask.shape[1])
    if h <= 0 or w <= 0:
        return

    out_region = out_region[:h, :w]
    img2_rect = img2_rect[:h, :w]
    mask = mask[:h, :w]

    out_region = out_region * (1 - mask) + img2_rect * mask
    out[r2[1]:r2[1] + h, r2[0]:r2[0] + w] = out_region


# 役割: 全三角形をワープして画像全体を変形
def warp_image(img, src_points, dst_points, tri_indices):
    """三角形ごとに変形して画像全体をワープする。"""
    out = img.copy().astype(np.float32)
    for i0, i1, i2 in tri_indices:
        t_src = [src_points[i0], src_points[i1], src_points[i2]]
        t_dst = [dst_points[i0], dst_points[i1], dst_points[i2]]
        warp_triangle(img, out, t_src, t_dst)
    return np.clip(out, 0, 255).astype(np.uint8)

# カメラを初期化
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

last_print = 0.0
frame_skip = 1  # 1なら毎フレーム処理。負荷を下げるなら2以上
frame_count = 0

# キャリブレーション基準値
baseline_brow_height = None  # 眉の上下位置（鼻の下基準、正規化y）
baseline_mouth_height = None  # 口角高さ（鼻の下基準、正規化y）

ensure_model(MODEL_PATH, MODEL_URL)

# Face Landmarker の初期化
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

landmarker = vision.FaceLandmarker.create_from_options(options)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_skip > 1 and (frame_count % frame_skip) != 0:
            # スキップフレームはそのまま表示してUIだけ維持
            cv2.imshow("Face Landmarks", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        # MediaPipe用にRGBへ変換
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = results.face_landmarks[0]
            display_frame = frame

            # 眉頭（内側）を動的に選ぶ
            left_brow_idx = max(
                BROW_LEFT_CANDIDATES, key=lambda i: face_landmarks[i].x
            )
            right_brow_idx = min(
                BROW_RIGHT_CANDIDATES, key=lambda i: face_landmarks[i].x
            )

            # 眉の上下位置と口角高さ（鼻の下基準）を計算
            left_brow = face_landmarks[left_brow_idx]
            right_brow = face_landmarks[right_brow_idx]

            nose_base = face_landmarks[NOSE_BASE_INDEX]
            brow_y = (left_brow.y + right_brow.y) / 2.0
            # 鼻の下から眉までの相対高さ（値が大きいほど眉が高い）
            brow_height = nose_base.y - brow_y
            mouth_left = face_landmarks[KEYPOINTS["mouth_left"]]
            mouth_right = face_landmarks[KEYPOINTS["mouth_right"]]
            mouth_y = (mouth_left.y + mouth_right.y) / 2.0
            # 鼻の下から口角までの相対高さ（値が大きいほど口角が高い）
            mouth_height = nose_base.y - mouth_y

            # 指定点のみ描画して負荷を最小化
            for name, idx in KEYPOINTS.items():
                lm = face_landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(display_frame, (x, y), 4, (0, 255, 255), -1)  # 黄

            for idx in (left_brow_idx, right_brow_idx):
                lm = face_landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(display_frame, (x, y), 4, (255, 0, 255), -1)  # 紫

            # キャリブレーションがあれば偏差を計算して表示・補正
            if baseline_brow_height is not None and baseline_mouth_height is not None:
                brow_delta = baseline_brow_height - brow_height
                mouth_delta = baseline_mouth_height - mouth_height

                # マイナス方向（眉が下がる / 口角が下がる）の変化のみ対象
                brow_neg = max(0.0, brow_delta)
                mouth_neg = max(0.0, mouth_delta)

                cv2.putText(
                    display_frame,
                    f"brow_delta: {-brow_neg:.4f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"mouth_delta: {-mouth_neg:.4f}",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )

                # 50%キャンセルした理想補正座標を計算
                # 眉: 鼻の下基準で高さを上げる
                corrected_brow_height = brow_height + (brow_neg * 0.5)
                corr_brow_y = nose_base.y - corrected_brow_height

                # 口角: 鼻の下基準で高さを上げる
                corrected_mouth_height = mouth_height + (mouth_neg * 0.5)
                corr_mouth_y = nose_base.y - corrected_mouth_height

                # 変形強度を適用（元座標との差分ベクトルを拡大）
                corr_brow_y = brow_y + (corr_brow_y - brow_y) * DEFORM_STRENGTH
                corr_mouth_y = mouth_y + (corr_mouth_y - mouth_y) * DEFORM_STRENGTH

                # 補正点を別色で描画（青）
                clx, cly = int(left_brow.x * w), int(corr_brow_y * h)
                crx, cry = int(right_brow.x * w), int(corr_brow_y * h)
                cv2.circle(display_frame, (clx, cly), 4, (255, 128, 0), -1)
                cv2.circle(display_frame, (crx, cry), 4, (255, 128, 0), -1)

                mlx, mrx = int(mouth_left.x * w), int(mouth_right.x * w)
                my = int(corr_mouth_y * h)
                cv2.circle(display_frame, (mlx, my), 4, (255, 128, 0), -1)
                cv2.circle(display_frame, (mrx, my), 4, (255, 128, 0), -1)

                # メッシュ変形のための対応点を作成
                src_points = landmarks_to_points(face_landmarks, w, h)
                dst_points = src_points.copy()

                dst_points[left_brow_idx][1] = corr_brow_y * h
                dst_points[right_brow_idx][1] = corr_brow_y * h
                dst_points[KEYPOINTS["mouth_left"]][1] = corr_mouth_y * h
                dst_points[KEYPOINTS["mouth_right"]][1] = corr_mouth_y * h

                # 変形点を間引く場合のインデックス
                if DEFORM_POINTS_STEP > 1:
                    indices = set(range(0, len(src_points), DEFORM_POINTS_STEP))
                    indices.update(
                        [left_brow_idx, right_brow_idx,
                         KEYPOINTS["mouth_left"], KEYPOINTS["mouth_right"]]
                    )
                    indices = sorted(indices)
                    src_points_sub = src_points[indices]
                    dst_points_sub = dst_points[indices]
                else:
                    src_points_sub = src_points
                    dst_points_sub = dst_points

                tri_indices = build_delaunay_triangles((0, 0, w, h), src_points_sub)
                display_frame = warp_image(frame, src_points_sub, dst_points_sub, tri_indices)

                # デバッグ表示: 元のランドマーク(赤)のみ
                for p in src_points_sub:
                    cv2.circle(display_frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

                # 眉頭は見えにくいので大きめに描画
                bl = (int(left_brow.x * w), int(left_brow.y * h))
                br = (int(right_brow.x * w), int(right_brow.y * h))
                cbl = (int(left_brow.x * w), int(corr_brow_y * h))
                cbr = (int(right_brow.x * w), int(corr_brow_y * h))
                cv2.circle(display_frame, bl, 5, (0, 0, 255), 2)
                cv2.circle(display_frame, br, 5, (0, 0, 255), 2)
                cv2.circle(display_frame, cbl, 5, (255, 128, 0), 2)
                cv2.circle(display_frame, cbr, 5, (255, 128, 0), 2)

            # 座標の出力は間引いて負荷を抑える
            now = time.time()
            if now - last_print >= PRINT_INTERVAL_SEC:
                last_print = now
                coords = {}
                for name, idx in KEYPOINTS.items():
                    lm = face_landmarks[idx]
                    coords[name] = (lm.x, lm.y, lm.z)

                coords["brow_inner_left"] = (left_brow.x, left_brow.y, left_brow.z)
                coords["brow_inner_right"] = (right_brow.x, right_brow.y, right_brow.z)
                print(coords)

        cv2.imshow("Face Landmarks", display_frame if results.face_landmarks else frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s") and results.face_landmarks:
            baseline_brow_height = brow_height
            baseline_mouth_height = mouth_height
            print("[Calibrated] brow_height=", baseline_brow_height,
                  "mouth_height=", baseline_mouth_height)
finally:
    # リソース解放
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
