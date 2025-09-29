# romp_test_full.py
import cv2
import torch
import romp
import argparse

# -------------------------------
# 1. 커맨드라인 파라미터
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["usb", "ip", "video"], default="usb",
                    help="Input mode: usb, ip, or video file")
parser.add_argument("--source", type=str, default=None,
                    help="IP camera URL or video file path")
args = parser.parse_args()

# -------------------------------
# 2. ROMP 모델 로드
# -------------------------------
romp_settings = romp.main.default_settings
romp_settings.mode = args.mode
romp_model = romp.ROMP(romp_settings)

# -------------------------------
# 3. 비디오/웹캠 입력 설정
# -------------------------------
if args.mode == "usb":
    cap = cv2.VideoCapture(0)  # 기본 USB 카메라
elif args.mode == "ip":
    if args.source is None:
        raise ValueError("IP camera mode requires --source URL")
    cap = cv2.VideoCapture(args.source)
elif args.mode == "video":
    if args.source is None:
        raise ValueError("Video mode requires --source file path")
    cap = cv2.VideoCapture(args.source)
else:
    raise ValueError("Invalid mode")

if not cap.isOpened():
    print("카메라/비디오 연결 실패")
    exit()

# -------------------------------
# 4. 실시간 SMPL 파라미터 추출
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # ROMP 추론
    with torch.no_grad():
        outputs = romp_model(frame)

    # AnimateDataset 형식으로 변환
    smpl_params = {}
    for key in ["global_orient", "body_pose", "transl"]:
        val = outputs.get(key, None)
        if val is None:
            smpl_params[key] = None
        else:
            # NumPy 배열이면 바로 list로, PyTorch tensor이면 .cpu().tolist()
            if torch.is_tensor(val):
                smpl_params[key] = val.cpu().tolist()
            else:
                smpl_params[key] = val.tolist()

    # 파라미터 크기 출력
    print(
        "global_orient:", len(smpl_params["global_orient"][0]) if smpl_params["global_orient"] else 0,
        "| body_pose:", len(smpl_params["body_pose"][0]) if smpl_params["body_pose"] else 0,
        "| transl:", len(smpl_params["transl"][0]) if smpl_params["transl"] else 0
    )

cap.release()
cv2.destroyAllWindows()
