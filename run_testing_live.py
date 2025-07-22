import socket
import json
import time
from gaussian_renderer import network_gui

def send_json(sock, data):
    msg = json.dumps(data).encode('utf-8')
    length = len(msg).to_bytes(4, 'little')
    sock.sendall(length + msg)

def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 6009))
    print("[run_testing_live] Connected to TCP server")

    network_gui.conn = sock  # 소켓을 network_gui.conn에 직접 넣음

    init_message = {
        "resolution_x": 640,
        "resolution_y": 480,
        "train": False,
        "fov_y": 45.0,
        "fov_x": 45.0,
        "z_near": 0.1,
        "z_far": 1000.0,
        "shs_python": False,
        "rot_scale_python": False,
        "keep_alive": True,
        "scaling_modifier": 1.0,
        "view_matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
        "view_projection_matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    }

    for i, _ in enumerate(frameset_test):
        print(f"[run_testing_live] Frame {i}")

        # 서버가 receive()를 호출하기 전에 클라이언트가 먼저 메시지를 보냄
        send_json(sock, init_message)

        # 메시지 전송 후 바로 render_to_network 호출 (서버가 메시지 받고 진행)
        network_gui.render_to_network(gs_model, pipe, verify if verify else "verify")

        time.sleep(0.033)

    sock.close()
    print("[run_testing_live] TCP connection closed")
