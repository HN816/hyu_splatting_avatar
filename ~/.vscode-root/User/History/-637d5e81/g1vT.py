import time
from gaussian_renderer import network_gui
import base64


def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    import time
    for i, _ in enumerate(frameset_test):
        # 렌더링과 네트워크 전송을 모두 처리하는 함수 호출
        network_gui.render_to_network(gs_model, pipe, verify if verify else "verify")
        # print(f"[run_testing_live] frame {i} sent")
        time.sleep(0.033)  # 30fps
