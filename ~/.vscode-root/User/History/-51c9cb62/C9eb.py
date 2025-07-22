#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
import numpy as np
import threading
import base64
import cv2
from scene.cameras import MiniCam

from flask import Flask, render_template
from flask_socketio import SocketIO

# TCP 서버 변수
host = "127.0.0.1"
port = 6009
conn = None
addr = None
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 6009))
listener.listen(1)


# Flask + SocketIO 서버 초기화
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')  # 아래 index.html 예제 참고

def start_web_server(ip, port_web):
    print(f"[network_gui] Starting Flask web server on http://{ip}:{port_web}")
    socketio.run(app, host=ip, port=port_web)

def init(wish_host, wish_port):
    global host, port, listener

    host = wish_host
    port = int(wish_port)

    try:
        if listener:
            listener.close()
    except Exception:
        pass

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 포트 재사용 허용

    print(f"[network_gui.init] binding to {host}:{port}")
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

    print(f"[network_gui] TCP server listening on {host}:{port}")

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except BlockingIOError:
        # 연결이 없으면 예외 무시하고 돌아가기
        pass
    except Exception as e:
        print(f"[network_gui] try_connect exception: {e}")


def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes is not None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    message = read()
    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None

########## TCP 통신 + 웹소켓 동시 지원 ##########


def send_image_to_network(image, verify):
    """
    TCP 클라이언트에게 이미지 전송 + 웹소켓 클라이언트에게 base64 인코딩 이미지 전송
    image: torch tensor (3,H,W), float [0,1]
    """
    global conn
    if conn is None:
        try_connect()
    if conn is not None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifier = receive()

            net_image = torch.zeros((3, custom_cam.image_height, custom_cam.image_width))
            if image.shape[1] > net_image.shape[1] or image.shape[2] > net_image.shape[2]:
                step = max(image.shape[1] / net_image.shape[1], image.shape[2] / net_image.shape[2])
                step = max(int(np.ceil(step)), 1)
                image = image[:, ::step, ::step]
            top = (net_image.shape[1] - image.shape[1]) // 2
            left = (net_image.shape[2] - image.shape[2]) // 2
            net_image[:, top:top+image.shape[1], left:left+image.shape[2]] = image

            # TCP 전송
            net_image_bytes = memoryview((torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)

            # --- 웹소켓 클라이언트 전송 ---
            np_img = (torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            # OpenCV로 JPEG 인코딩
            _, img_encoded = cv2.imencode('.jpg', np_img)
            jpg_as_text = base64.b64encode(img_encoded.tobytes()).decode()
            # Flask-SocketIO 이벤트 emit
            socketio.emit('new_frame', {'image': jpg_as_text})

        except Exception as e:
            print(f"[network_gui] send_image_to_network exception: {e}")
            conn = None

def render_to_network(model, pipe, verify, gt_image=None):
    """
    TCP 및 웹소켓으로 렌더링 이미지 전송
    """
    do_training = True
    global conn

    if conn is None:
        try_connect()
    if conn is not None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifier = receive()

            with torch.no_grad():
                net_image = model.render_to_camera(custom_cam, pipe, background='white',
                                                  scaling_modifer=scaling_modifier)["render"]

            if gt_image is not None:
                step = int(max(max(gt_image.shape[1] / 200, gt_image.shape[2] / 200), 1))
                img = gt_image[:, ::step, ::step]
                net_image[:, :img.shape[1], :img.shape[2]] = img

            # TCP 전송
            net_image_bytes = memoryview((torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)

            # 웹소켓 클라이언트 전송
            np_img = (torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            _, img_encoded = cv2.imencode('.jpg', np_img)
            jpg_as_text = base64.b64encode(img_encoded.tobytes()).decode()
            print(f"[network_gui] Emitting new_frame event, size: {len(jpg_as_text)}")

            socketio.emit('new_frame', {'image': jpg_as_text})

        except Exception as e:
            print(f"[network_gui] render_to_network exception: {e}")
            conn = None

    return do_training


# 웹서버 단독 실행 시
if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='127.0.0.1')
    parser.add_argument('--tcp_port', type=int, default=6009)
    parser.add_argument('--web_port', type=int, default=6010)
    args = parser.parse_args()

    init(args.ip, args.tcp_port)

    # Flask 웹서버 별도 쓰레드 실행
    threading.Thread(target=start_web_server, args=(args.ip, args.web_port), daemon=True).start()

    # TCP 서버 루프 (단순하게 연결만 유지, 실제 통신은 호출 함수에서 처리)
    print("[network_gui] TCP server ready")
    try:
        while True:
            try_connect()
    except KeyboardInterrupt:
        print("Shutting down server")
