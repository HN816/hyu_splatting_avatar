import time
from gaussian_renderer import network_gui

def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    for i, frame in enumerate(frameset_test):
        rendered_tensor = gs_model.render_frame(frame)  # (3,H,W), float tensor expected

        # 네트워크로 바로 전송 (TCP+웹소켓)
        network_gui.send_image_to_network(rendered_tensor, verify if verify else "verify")

        print(f"[run_testing_live] frame {i} sent")
        time.sleep(0.033)  # 30fps 맞춰서 조절
