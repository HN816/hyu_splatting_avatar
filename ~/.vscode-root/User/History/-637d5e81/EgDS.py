def run_testing_live(pipe, frameset_test, gs_model, pc_dir, verify=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 6009))
    print("[run_testing_live] Connected to TCP server")

    # 서버가 기대하는 JSON 메시지 보내기 (한 번만 보내면 됨)
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
    send_json(sock, init_message)
    print("[run_testing_live] Sent init JSON message")

    network_gui.conn = sock  # 네트워크 GUI에 소켓 할당

    for i, _ in enumerate(frameset_test):
        print("Frame", i)
        network_gui.render_to_network(gs_model, pipe, verify if verify else "verify")
        time.sleep(0.033)

    sock.close()
    print("[run_testing_live] TCP connection closed")


if __name__ == '__main__':
    dummy_frameset = range(100)  # 테스트용 프레임 100개
    run_testing_live(pipe=None, frameset_test=dummy_frameset, gs_model=your_model, pc_dir=None)
