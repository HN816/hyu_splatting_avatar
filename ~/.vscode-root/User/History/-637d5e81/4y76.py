import time
from gaussian_renderer import network_gui
import base64

import socket

def connect_to_tcp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 6009))
    return sock

sock = connect_to_tcp_server()


def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    # 🔧 TCP 클라이언트 연결을 기다림
    if network_gui.conn is None:
        print("[run_testing_live] Waiting for TCP client connection...")
        network_gui.try_connect()  # 블로킹 연결

    # 💡 이후 렌더링 프레임 전송
    for i, _ in enumerate(frameset_test):
        network_gui.render_to_network(gs_model, pipe, verify if verify else "verify")
        time.sleep(0.033)  # 30fps
