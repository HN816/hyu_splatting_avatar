import base64
import cv2
import numpy as np
import time

def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    for i, frame in enumerate(frameset_test):
        # 1. 프레임 렌더링 (예시)
        rendered_img_tensor = gs_model.render_to_camera(frame)  # (3,H,W), torch tensor 가정
        rendered_img = rendered_img_tensor.permute(1, 2, 0).cpu().numpy()  # (H,W,3) numpy

        # 2. [0~1] float → [0~255] uint8, RGB→BGR 변환 (OpenCV는 BGR)
        rendered_img_uint8 = (rendered_img * 255).astype(np.uint8)
        rendered_img_bgr = cv2.cvtColor(rendered_img_uint8, cv2.COLOR_RGB2BGR)

        # 3. JPEG 인코딩
        success, img_encoded = cv2.imencode('.jpg', rendered_img_bgr)
        if not success:
            print(f"[run_testing_live] JPEG encoding failed at frame {i}")
            continue

        # 4. base64 인코딩
        jpg_as_text = base64.b64encode(img_encoded.tobytes()).decode()
        print(f"[run_testing_live] Frame {i} image size: {len(jpg_as_text)}")

        # TODO: 이 jpg_as_text 를 네트워크 등으로 전송하는 코드 추가

        time.sleep(0.033)  # 30fps 조절
