import time
from PIL import Image
import numpy as np

def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    for i, frame in enumerate(frameset_test):
        # 여기는 실제 렌더링 부분, gs_model.render_frame 예시 가정
        # render_frame() 함수가 PIL 이미지 반환한다고 가정
        rendered_image = gs_model.render_frame(frame)  

        # 이미지 실시간 전송
        from gaussian_renderer import network_gui
        network_gui.send_image(rendered_image)

        print(f'[frame {i}] Sent live image')
        time.sleep(0.033)  # 약 30fps 속도 조절 (필요에 따라 조정)
