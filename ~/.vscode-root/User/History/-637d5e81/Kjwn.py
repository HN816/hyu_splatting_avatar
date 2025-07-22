import time
from gaussian_renderer import network_gui
import base64


def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    # 🔧 TCP 서버 연결 수락
    if network_gui.conn is None:
        print("[run_testing_live] Waiting for TCP client connection...")
        network_gui.try_connect()

    for i, _ in enumerate(frameset_test):
        network_gui.render_to_network(gs_model, pipe, verify if verify else "verify")
        time.sleep(0.033)
