import cv2
import torch
from simple_romp import SimpleROMP

# 1. 모델 로드
model = SimpleROMP().cuda().eval()

# 2. 웹캠 입력
IP_CAM_URL = "http://192.0.0.5:8080/video"
cap = cv2.VideoCapture(IP_CAM_URL)
if not cap.isOpened():
    print("카메라 연결 실패. IP 주소 확인 필요")
    exit()

# 3. 실시간 SMPL 파라미터 확인
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    with torch.no_grad():
        results = model(frame)

    # AnimateDataset 형식 변환: tensor → 리스트
    smpl_params = {
        "global_orient": results["global_orient"].cpu().tolist(),
        "body_pose": results["body_pose"].cpu().tolist(),
        "transl": results["transl"].cpu().tolist(),
    }

    # 파라미터 사이즈 출력
    print(
        "global_orient:", len(smpl_params["global_orient"][0]),
        "| body_pose:", len(smpl_params["body_pose"][0]),
        "| transl:", len(smpl_params["transl"][0])
    )

    # 화면 표시
    cv2.imshow("IP Webcam - Simple ROMP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
