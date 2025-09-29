import cv2
import torch
import romp
from flask import Flask, jsonify
from threading import Thread, Lock
import time

# -------------------------------
# Flask 서버
# -------------------------------
app = Flask(__name__)
pose_lock = Lock()
latest_pose = None  # 마지막 추출 SMPL 파라미터

# -------------------------------
# ROMP 모델 로드
# -------------------------------
romp_settings = romp.main.default_settings
romp_settings.mode = "video"  # USB/웹캠용
romp_model = romp.ROMP(romp_settings)
print("[INFO] ROMP 모델 로드 완료", flush=True)

# -------------------------------
# USB/웹캠 입력
# -------------------------------
USB_CAM_INDEX = 0
cap = cv2.VideoCapture(USB_CAM_INDEX)
if not cap.isOpened():
    print("[ERROR] 카메라 연결 실패", flush=True)
    exit()
print(f"[INFO] USB 카메라 {USB_CAM_INDEX} 연결 완료", flush=True)

# -------------------------------
# 실시간 ROMP 루프
# -------------------------------
def romp_loop():
    global latest_pose
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패, 재시도...", flush=True)
            time.sleep(0.05)
            continue

        with torch.no_grad():
            outputs = romp_model(frame)

        if outputs is None:
            # 사람 감지 안되면 건너뜀
            time.sleep(0.05)
            continue

        # AnimateDataset 형식 변환
        smpl_params = {}
        for key in ["global_orient", "body_pose", "transl"]:
            val = outputs.get(key, None)
            if val is None:
                if key == "transl":
                    smpl_params[key] = [0, 0.15, 5] 
                    print("[WARN] transl이 None -> [0,0.15,5]로 대체", flush=True)
                else:
                    smpl_params[key] = None
            else:
                if torch.is_tensor(val):
                    smpl_params[key] = val.cpu().tolist()
                else:
                    smpl_params[key] = val.tolist()

        # log shapes
        print(
            f"[DEBUG] pose_params shapes -> "
            f"global_orient: {torch.tensor(smpl_params['global_orient']).shape} | "
            f"body_pose: {torch.tensor(smpl_params['body_pose']).shape} | "
            f"transl: {torch.tensor(smpl_params['transl']).shape if smpl_params['transl'] is not None else 'None'}",
            flush=True
        )

        # 최신 파라미터 업데이트 (쓰레드 안전)
        with pose_lock:
            latest_pose = smpl_params

        time.sleep(0.05)  # 약 20 FPS

# -------------------------------
# Flask 라우트
# -------------------------------
@app.route("/pose/latest", methods=["GET"])
def get_latest_pose():
    with pose_lock:
        if latest_pose is None:
            return jsonify({"end": True, "message": "No person detected yet"})
        return jsonify(latest_pose)

# -------------------------------
# 메인
# -------------------------------
if __name__ == "__main__":
    print("[INFO] ROMP 루프 쓰레드 시작", flush=True)
    t = Thread(target=romp_loop, daemon=True)
    t.start()

    print("[INFO] Flask 서버 시작: http://0.0.0.0:9000/pose/latest", flush=True)
    # debug=True: 로그 항상 표시, use_reloader=False: 중복 실행 방지
    app.run(host="0.0.0.0", port=9000, debug=True, use_reloader=False)
