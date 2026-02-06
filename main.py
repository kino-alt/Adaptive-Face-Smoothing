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

# 眉ラインの候補点（左/右）
# この集合から「眉頭・中央・眉尻」を動的に選ぶ
BROW_LEFT_LINE = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_RIGHT_LINE = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# 眉周辺の皮膚を連動させるための領域点（必要に応じて調整）
BROW_LEFT_REGION = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 156, 143, 124, 46]
BROW_RIGHT_REGION = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276, 383, 372, 353, 276]

# 鼻の下（基準点）として使うランドマーク
# 必要なら別番号に変更してください
NOSE_BASE_INDEX = 2

# 低負荷で動かすためのキャプチャ設定
FRAME_W, FRAME_H = 640, 480
PRINT_INTERVAL_SEC = 0.2  # コンソール出力の間隔（秒）
DEFORM_POINTS_STEP = 1  # 変形用の点数を間引くなら2以上
DEFORM_STRENGTH = 1.2  # 変形強度(1.0〜3.0)
MOUTH_LIFT_PX = 3  # 口角を常時リフトする固定オフセット(px)
BROW_SMOOTH_FACTOR = -0.8  # 眉を下げた移動量への係数（上方向へ押し戻す）
DEBUG_LINE_SCALE = 10.0  # 補正ラインを見やすくする倍率
REMAP_RADIUS_PX = 30  # remapで動かす半径（眉用）
REMAP_SIGMA_PX = 10.0  # ガウス重みのsigma（眉用）
MOUTH_RADIUS_PX = 40  # 口角周辺の影響半径
MOUTH_SIGMA_PX = 16.0  # 口角のガウス減衰
CHEEK_LIFT_RATIO = 0.2  # 口角リフトに対する頬の持ち上げ比率
MOUTH_CENTER_IDS = [13, 14]  # 唇中央付近（固定したい点）
MOUTH_CENTER_RADIUS_PX = 22  # 口中央の固定領域半径
MOUTH_CENTER_SIGMA_PX = 8.0  # 口中央の固定用ガウス
MOUTH_CENTER_STRENGTH = 0.3  # 口中央の固定強度（小さいほど中心が動く）
REMAP_Y_SIGN = -1.0  # Y方向の符号補正
CORNER_SPREAD_RATIO = 0.12  # 口角の外側への広がり率
LIP_CURVE_BLEND = 0.07 # 唇ラインをベジェ曲線へ寄せる強さ
LIP_RADIUS_PX = 10  # 唇ラインの影響半径
LIP_SIGMA_PX = 4.0  # 唇ラインのガウス減衰

# 口の中心/ライン用ランドマーク
UPPER_LIP_CENTER_ID = 0
LOWER_LIP_CENTER_ID = 17
UPPER_LIP_IDS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_IDS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

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
calibrated_view = False  # デバッグ表示用（通常はFalse）
apply_correction = True
grid_x = None
grid_y = None
grid_size = None

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
        display_frame = frame
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
            line_points = []

            # remap用のグリッドを準備
            if grid_size != (h, w):
                grid_x, grid_y = np.meshgrid(
                    np.arange(w, dtype=np.float32),
                    np.arange(h, dtype=np.float32)
                )
                grid_size = (h, w)

            # 眉頭・中央・眉尻を動的に選ぶ
            left_sorted = sorted(BROW_LEFT_LINE, key=lambda i: face_landmarks[i].x)
            left_brow_inner_idx = left_sorted[-2] if len(left_sorted) >= 2 else left_sorted[-1]
            left_brow_tail_idx = min(
                BROW_LEFT_LINE, key=lambda i: face_landmarks[i].x
            )
            left_mid_target = (
                face_landmarks[left_brow_inner_idx].x +
                face_landmarks[left_brow_tail_idx].x
            ) / 2.0
            left_brow_mid_idx = min(
                BROW_LEFT_LINE, key=lambda i: abs(face_landmarks[i].x - left_mid_target)
            )

            right_sorted = sorted(BROW_RIGHT_LINE, key=lambda i: face_landmarks[i].x)
            right_brow_inner_idx = right_sorted[1] if len(right_sorted) >= 2 else right_sorted[0]
            right_brow_tail_idx = max(
                BROW_RIGHT_LINE, key=lambda i: face_landmarks[i].x
            )
            right_mid_target = (
                face_landmarks[right_brow_inner_idx].x +
                face_landmarks[right_brow_tail_idx].x
            ) / 2.0
            right_brow_mid_idx = min(
                BROW_RIGHT_LINE, key=lambda i: abs(face_landmarks[i].x - right_mid_target)
            )

            # 眉の上下位置と口角高さ（鼻の下基準）を計算
            left_brow = face_landmarks[left_brow_inner_idx]
            right_brow = face_landmarks[right_brow_inner_idx]
            left_brow_mid = face_landmarks[left_brow_mid_idx]
            right_brow_mid = face_landmarks[right_brow_mid_idx]
            left_brow_tail = face_landmarks[left_brow_tail_idx]
            right_brow_tail = face_landmarks[right_brow_tail_idx]

            nose_base = face_landmarks[NOSE_BASE_INDEX]
            brow_y = (
                left_brow.y + left_brow_mid.y + left_brow_tail.y +
                right_brow.y + right_brow_mid.y + right_brow_tail.y
            ) / 6.0
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

            for idx in (left_brow_inner_idx, right_brow_inner_idx):
                lm = face_landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(display_frame, (x, y), 4, (255, 0, 255), -1)  # 紫

            # 補正座標の初期値（未補正）
            corr_left_inner_y = left_brow.y
            corr_left_mid_y = left_brow_mid.y
            corr_left_tail_y = left_brow_tail.y
            corr_right_inner_y = right_brow.y
            corr_right_mid_y = right_brow_mid.y
            corr_right_tail_y = right_brow_tail.y
            corr_mouth_y = mouth_y
            brow_offset = 0.0
            do_warp = apply_correction

            # 口角は常に固定オフセットでリフトアップ（補正ON時）
            if apply_correction:
                corr_mouth_y = mouth_y - (MOUTH_LIFT_PX / float(h))

            # キャリブレーションがあれば眉の平滑化を計算
            if apply_correction and baseline_brow_height is not None:
                # 眉が下がった移動量を検知（基準より高さが減った分）
                down_amount = max(0.0, baseline_brow_height - brow_height)
                # 下がった分にマイナス係数を掛けて上方向へ押し戻す
                corr_brow_y = brow_y + (down_amount * BROW_SMOOTH_FACTOR)

                # 変形強度を適用（元座標との差分ベクトルを拡大）
                corr_brow_y = brow_y + (corr_brow_y - brow_y) * DEFORM_STRENGTH

                # 眉は形状を維持するため、全点に同じオフセットを適用
                brow_offset = corr_brow_y - brow_y
                corr_left_inner_y = left_brow.y + brow_offset
                corr_left_mid_y = left_brow_mid.y + brow_offset
                corr_left_tail_y = left_brow_tail.y + brow_offset
                corr_right_inner_y = right_brow.y + brow_offset
                corr_right_mid_y = right_brow_mid.y + brow_offset
                corr_right_tail_y = right_brow_tail.y + brow_offset

                # 補正量の可視化（通常表示時のみ）
                if not calibrated_view:
                    cv2.putText(
                        display_frame,
                        f"brow_down: {down_amount:.4f}",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 200),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"brow_offset: {brow_offset:.4f}",
                        (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 200),
                        2,
                    )

                # 眉の元位置→補正位置のラインを後で描画
                line_points = [
                    (left_brow.x, left_brow.y, corr_left_inner_y),
                    (right_brow.x, right_brow.y, corr_right_inner_y),
                ]

                # 補正点の描画はワープ後に行う
            else:
                if not calibrated_view:
                    cv2.putText(
                        display_frame,
                        "Press K to calibrate",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            if do_warp:
                # new_landmarks と original_landmarks を作成（ピクセル座標）
                original_landmarks = landmarks_to_points(face_landmarks, w, h)
                new_landmarks = original_landmarks.copy()

                new_landmarks[left_brow_inner_idx][1] = corr_left_inner_y * h
                new_landmarks[left_brow_mid_idx][1] = corr_left_mid_y * h
                new_landmarks[left_brow_tail_idx][1] = corr_left_tail_y * h
                new_landmarks[right_brow_inner_idx][1] = corr_right_inner_y * h
                new_landmarks[right_brow_mid_idx][1] = corr_right_mid_y * h
                new_landmarks[right_brow_tail_idx][1] = corr_right_tail_y * h
                new_landmarks[KEYPOINTS["mouth_left"]][1] = corr_mouth_y * h
                new_landmarks[KEYPOINTS["mouth_right"]][1] = corr_mouth_y * h

                # 口角を外側にも広げる（移動量の20%）
                corner_dy_px = abs((corr_mouth_y - mouth_y) * h)
                corner_dx_px = corner_dy_px * CORNER_SPREAD_RATIO
                new_landmarks[KEYPOINTS["mouth_left"]][0] = original_landmarks[KEYPOINTS["mouth_left"]][0] - corner_dx_px
                new_landmarks[KEYPOINTS["mouth_right"]][0] = original_landmarks[KEYPOINTS["mouth_right"]][0] + corner_dx_px

                # 頬（口角上部）も30%持ち上げる
                cheek_dy = corner_dy_px * CHEEK_LIFT_RATIO
                for cheek_id in (205, 425):
                    new_landmarks[cheek_id][1] = original_landmarks[cheek_id][1] - cheek_dy

                # 二次ベジェ曲線で唇ラインを再配置
                upper_center = original_landmarks[UPPER_LIP_CENTER_ID]
                lower_center = original_landmarks[LOWER_LIP_CENTER_ID]
                left_corner = new_landmarks[KEYPOINTS["mouth_left"]]
                right_corner = new_landmarks[KEYPOINTS["mouth_right"]]

                mouth_center_x = (upper_center[0] + lower_center[0]) / 2.0
                left_x = original_landmarks[KEYPOINTS["mouth_left"]][0]
                right_x = original_landmarks[KEYPOINTS["mouth_right"]][0]

                for idx in set(UPPER_LIP_IDS + LOWER_LIP_IDS):
                    if idx in (UPPER_LIP_CENTER_ID, LOWER_LIP_CENTER_ID):
                        continue
                    px = original_landmarks[idx][0]
                    if px <= mouth_center_x:
                        denom = max(mouth_center_x - left_x, 1.0)
                        t = np.clip((mouth_center_x - px) / denom, 0.0, 1.0)
                        p0 = upper_center
                        p1 = left_corner
                        p2 = lower_center
                    else:
                        denom = max(right_x - mouth_center_x, 1.0)
                        t = np.clip((px - mouth_center_x) / denom, 0.0, 1.0)
                        p0 = upper_center
                        p1 = right_corner
                        p2 = lower_center

                    # Quadratic Bezier
                    bez_y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]

                    # 下唇は皿状になるように滑らかに補間
                    if idx in LOWER_LIP_IDS:
                        blend = LIP_CURVE_BLEND * (0.6 + 0.4 * t)
                    else:
                        blend = LIP_CURVE_BLEND

                    new_landmarks[idx][1] = (1 - blend) * original_landmarks[idx][1] + blend * bez_y

                # 影響範囲を限定した remap を作成
                map_x = grid_x.copy()
                map_y = grid_y.copy()

                centers = [
                    (left_brow_inner_idx, REMAP_RADIUS_PX, REMAP_SIGMA_PX),
                    (left_brow_mid_idx, REMAP_RADIUS_PX, REMAP_SIGMA_PX),
                    (left_brow_tail_idx, REMAP_RADIUS_PX, REMAP_SIGMA_PX),
                    (right_brow_inner_idx, REMAP_RADIUS_PX, REMAP_SIGMA_PX),
                    (right_brow_mid_idx, REMAP_RADIUS_PX, REMAP_SIGMA_PX),
                    (right_brow_tail_idx, REMAP_RADIUS_PX, REMAP_SIGMA_PX),
                    (KEYPOINTS["mouth_left"], MOUTH_RADIUS_PX, MOUTH_SIGMA_PX),
                    (KEYPOINTS["mouth_right"], MOUTH_RADIUS_PX, MOUTH_SIGMA_PX),
                    (205, MOUTH_RADIUS_PX, MOUTH_SIGMA_PX),
                    (425, MOUTH_RADIUS_PX, MOUTH_SIGMA_PX),
                ]
                for lip_id in set(UPPER_LIP_IDS + LOWER_LIP_IDS):
                    centers.append((lip_id, LIP_RADIUS_PX, LIP_SIGMA_PX))

                for idx, radius_px, sigma_px in centers:
                    ox, oy = original_landmarks[idx]
                    nx, ny = new_landmarks[idx]
                    # remapは「出力画素が参照する入力座標」を指定するため、
                    # 変形は逆方向（元→新の差分を反転）で与える
                    dx = ox - nx
                    dy = (oy - ny) * REMAP_Y_SIGN
                    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                        continue

                    x0 = max(int(ox - radius_px), 0)
                    x1 = min(int(ox + radius_px), w - 1)
                    y0 = max(int(oy - radius_px), 0)
                    y1 = min(int(oy + radius_px), h - 1)
                    if x1 <= x0 or y1 <= y0:
                        continue

                    roi_x = grid_x[y0:y1, x0:x1] - ox
                    roi_y = grid_y[y0:y1, x0:x1] - oy
                    weight = np.exp(-(roi_x ** 2 + roi_y ** 2) / (2.0 * sigma_px ** 2))

                    map_x[y0:y1, x0:x1] += weight * dx
                    map_y[y0:y1, x0:x1] += weight * dy

                # 口中央を固定するため、口角の平均移動量を打ち消す
                mouth_dx = (
                    (original_landmarks[KEYPOINTS["mouth_left"]][0] - new_landmarks[KEYPOINTS["mouth_left"]][0]) +
                    (original_landmarks[KEYPOINTS["mouth_right"]][0] - new_landmarks[KEYPOINTS["mouth_right"]][0])
                ) / 2.0
                mouth_dy = (
                    (original_landmarks[KEYPOINTS["mouth_left"]][1] - new_landmarks[KEYPOINTS["mouth_left"]][1]) +
                    (original_landmarks[KEYPOINTS["mouth_right"]][1] - new_landmarks[KEYPOINTS["mouth_right"]][1])
                ) / 2.0
                mouth_dy *= REMAP_Y_SIGN

                for center_id in MOUTH_CENTER_IDS:
                    ox, oy = original_landmarks[center_id]
                    x0 = max(int(ox - MOUTH_CENTER_RADIUS_PX), 0)
                    x1 = min(int(ox + MOUTH_CENTER_RADIUS_PX), w - 1)
                    y0 = max(int(oy - MOUTH_CENTER_RADIUS_PX), 0)
                    y1 = min(int(oy + MOUTH_CENTER_RADIUS_PX), h - 1)
                    if x1 <= x0 or y1 <= y0:
                        continue

                    roi_x = grid_x[y0:y1, x0:x1] - ox
                    roi_y = grid_y[y0:y1, x0:x1] - oy
                    weight = np.exp(-(roi_x ** 2 + roi_y ** 2) / (2.0 * MOUTH_CENTER_SIGMA_PX ** 2))
                    map_x[y0:y1, x0:x1] += weight * (-mouth_dx) * MOUTH_CENTER_STRENGTH
                    map_y[y0:y1, x0:x1] += weight * (-mouth_dy) * MOUTH_CENTER_STRENGTH

                # remapで変形を反映
                display_frame = cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

            # キャリブレーション表示（デバッグ用）
            if calibrated_view:
                cv2.circle(display_frame, (int(left_brow.x * w), int(corr_left_inner_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(left_brow_mid.x * w), int(corr_left_mid_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(left_brow_tail.x * w), int(corr_left_tail_y * h)), 6, (255, 0, 0), -1)

                cv2.circle(display_frame, (int(right_brow.x * w), int(corr_right_inner_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(right_brow_mid.x * w), int(corr_right_mid_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(right_brow_tail.x * w), int(corr_right_tail_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(mouth_left.x * w), int(corr_mouth_y * h)), 6, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(mouth_right.x * w), int(corr_mouth_y * h)), 6, (255, 0, 0), -1)

            # 眉の元位置→補正位置を線で表示（ワープ後に描画）
            if line_points:
                for (sx, sy, cy) in line_points:
                    # 視認性のためラインを拡大表示（描画のみ）
                    dy = (cy - sy) * DEBUG_LINE_SCALE
                    if abs(dy) < 0.005:
                        dy = 0.02 if dy >= 0 else -0.02
                    y2 = min(max(sy + dy, 0.0), 1.0)
                    p1 = (int(sx * w), int(sy * h))
                    p2 = (int(sx * w), int(y2 * h))
                    cv2.line(display_frame, p1, p2, (0, 255, 0), 3)
                    cv2.circle(display_frame, p1, 4, (0, 255, 0), -1)
                    cv2.circle(display_frame, p2, 4, (0, 255, 0), -1)

            # 補正点をワープ後に描画（実際の位置確認用）
            if apply_correction and baseline_brow_height is not None:
                for (sx, sy, cy) in [
                    (left_brow.x, left_brow.y, corr_left_inner_y),
                    (left_brow_mid.x, left_brow_mid.y, corr_left_mid_y),
                    (left_brow_tail.x, left_brow_tail.y, corr_left_tail_y),
                    (right_brow.x, right_brow.y, corr_right_inner_y),
                    (right_brow_mid.x, right_brow_mid.y, corr_right_mid_y),
                    (right_brow_tail.x, right_brow_tail.y, corr_right_tail_y),
                ]:
                    cv2.circle(display_frame, (int(sx * w), int(sy * h)), 3, (0, 0, 255), 1)
                    cv2.circle(display_frame, (int(sx * w), int(cy * h)), 4, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(mouth_left.x * w), int(mouth_left.y * h)), 3, (0, 0, 255), 1)
                cv2.circle(display_frame, (int(mouth_right.x * w), int(mouth_right.y * h)), 3, (0, 0, 255), 1)
                cv2.circle(display_frame, (int(mouth_left.x * w), int(corr_mouth_y * h)), 4, (255, 0, 0), -1)
                cv2.circle(display_frame, (int(mouth_right.x * w), int(corr_mouth_y * h)), 4, (255, 0, 0), -1)

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

        # 1画面のみ表示（cキーで補正ON/OFF切り替え）
        view_frame = display_frame if apply_correction else frame
        status_text = "Correction: ON" if apply_correction else "Correction: OFF"
        cv2.putText(
            view_frame,
            status_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            view_frame,
            "C: toggle  K: calibrate",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.imshow("Face Landmarks", view_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("c"):
            apply_correction = not apply_correction
        if key == ord("v"):
            calibrated_view = not calibrated_view
        if key == ord("k") and results.face_landmarks:
            baseline_brow_height = brow_height
            baseline_mouth_height = mouth_height
            print("[Calibrated] brow_height=", baseline_brow_height,
                  "mouth_height=", baseline_mouth_height)
finally:
    # リソース解放
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
