import torch
import traceback
import socket
import json
import numpy as np
from scene.cameras import MiniCam
from PIL import Image
import io

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def tensor_to_png_bytes(tensor):
    # tensor shape: (3, H, W), 값 범위 [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3), uint8
    img = Image.fromarray(array)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        png_bytes = output.getvalue()
    return png_bytes

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\n[try_connect] 클라이언트 연결 수락됨: {addr}")
        conn.settimeout(None)

        # 서버가 최초 연결 직후 클라이언트에 초기 JSON 메시지를 보냄
        initial_message = {
            "resolution_x": 1080,
            "resolution_y": 1080,
            "train": True,
            "fov_y": 45.0,
            "fov_x": 60.0,
            "z_near": 0.1,
            "z_far": 100.0,
            "shs_python": False,
            "rot_scale_python": False,
            "keep_alive": True,
            "scaling_modifier": 1.0,
            # 예시용 view matrix (4x4 identity)
            "view_matrix": np.identity(4).flatten().tolist(),
            "view_projection_matrix": np.identity(4).flatten().tolist(),
        }
        send_json(initial_message)
    except Exception as inst:
        pass

def send_json(message_dict):
    global conn
    try:
        message_str = json.dumps(message_dict)
        message_bytes = message_str.encode('utf-8')
        length_bytes = len(message_bytes).to_bytes(4, 'little')
        conn.sendall(length_bytes)
        conn.sendall(message_bytes)
        print(f"[send_json] 메시지 전송 완료, 길이: {len(message_bytes)}")
    except Exception as e:
        print(f"[send_json] 예외 발생: {e}")
        traceback.print_exc()
        conn = None

def read():
    global conn
    messageLength = conn.recv(4)
    print(f"[read] Length raw: {messageLength}")
    if not messageLength or len(messageLength) < 4:
        raise ConnectionError("메시지 길이 수신 실패 또는 연결 종료")
    
    messageLength = int.from_bytes(messageLength, 'little')
    print(f"[read] Expecting message of length: {messageLength}")
    
    message = b''
    while len(message) < messageLength:
        chunk = conn.recv(messageLength - len(message))
        print(f"[read] Received chunk of size {len(chunk)}")
        if not chunk:
            raise ConnectionError("메시지 수신 중 연결 종료")
        message += chunk
    
    decoded = message.decode("utf-8")
    print(f"[read] Full message: {decoded}")
    
    return json.loads(decoded)

def send(message_bytes, verify):
    global conn
    if message_bytes is not None:
        length_bytes = len(message_bytes).to_bytes(4, 'little')  # 이미지 바이트 길이 전송
        conn.sendall(length_bytes)
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))              # verify 문자열 길이 전송
    conn.sendall(bytes(verify, 'ascii'))                         # verify 문자열 전송

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
            print(f"[receive] custom_cam: {custom_cam}, do_training: {do_training}, scaling_modifier: {scaling_modifier}")

        except Exception as e:
            print("receive exception")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None

########## routine ##########
def send_image_to_network(image, verify):
    global conn
    if conn == None:
        try_connect()
    if conn != None:
        try:
            custom_cam, do_training, do_shs_python, _, _, scaling_modifier = receive()

            net_image = torch.zeros((3, custom_cam.image_height, custom_cam.image_width))
            if image.shape[1] > net_image.shape[1] or image.shape[2] > net_image.shape[2]:
                step = max(image.shape[1] / net_image.shape[1], image.shape[2] / net_image.shape[2])
                step = max(int(np.ceil(step)), 1)
                image = image[:3, ::step, ::step]
            top = (net_image.shape[1] - image.shape[1]) // 2
            left = (net_image.shape[2] - image.shape[2]) // 2
            net_image[:, top:top+image.shape[1], left:left+image.shape[2]] = image

            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            send(net_image_bytes, verify)
        except Exception as e: 
            print(f"[send_image_to_network] 예외 발생: {e}")
            traceback.print_exc()
            conn = None

def render_to_network(model, pipe, verify, gt_image=None):
    do_training = True
    global conn

    if conn == None:
        try_connect()
    if conn != None:
        try:
            print("[render_to_network] receive() 호출")
            custom_cam, do_training, do_shs_python, _, _, scaling_modifier = receive()

            with torch.no_grad():
                net_image = model.render_to_camera(custom_cam, pipe, background='white',
                                                    scaling_modifier=scaling_modifier)["render"]
                print(f"[render_to_network] Received custom_cam: {custom_cam}, do_training: {do_training}, scaling_modifier: {scaling_modifier}")

            if gt_image is not None:
                print(f"[render_to_network] GT 이미지 존재, shape: {gt_image.shape}")
                # step = int(max(max(gt_image.shape[1] / 200, gt_image.shape[2] / 200), 1))
                step = 1
                img = gt_image[:, ::step, ::step]
                net_image[:, :img.shape[1], :img.shape[2]] = img

            net_image_bytes = tensor_to_png_bytes(net_image)

            send(net_image_bytes, verify)

        except Exception as e: 
            print(f"[render_to_network] Exception: {e}")
            traceback.print_exc()
            conn = None
    
    return do_training
