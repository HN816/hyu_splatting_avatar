import socket
import json
import io
from PIL import Image
import requests
import traceback

HOST = '127.0.0.1'
PORT = 6009
FLASK_SERVER_URL = 'http://127.0.0.1:8000/upload_image'
verify = "verify"

def recv_all(sock, length):
    data = b''
    try:
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
    except Exception as e:
        traceback.print_exc()
        return None
    return data

def read_json(sock):
    length_bytes = recv_all(sock, 4)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'little')
    message_bytes = recv_all(sock, length)
    if not message_bytes:
        return None
    return json.loads(message_bytes.decode('utf-8'))

def send_json(sock, message_dict):
    try:
        message_str = json.dumps(message_dict)
        message_bytes = message_str.encode('utf-8')
        length_bytes = len(message_bytes).to_bytes(4, 'little')
        sock.sendall(length_bytes)
        sock.sendall(message_bytes)
    except Exception as e:
        traceback.print_exc()

def read_image(sock):
    length_bytes = recv_all(sock, 4)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'little')
    img_bytes = recv_all(sock, length)
    return img_bytes

def send_verify(sock):
    try:
        sock.sendall(len(verify).to_bytes(4, 'little'))
        sock.sendall(verify.encode('ascii'))
    except Exception as e:
        traceback.print_exc()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
    except Exception as e:
        return

    # 최초 연결 시 JSON 메시지 수신 및 저장
    initial_json = read_json(sock)
    if initial_json is None:
        return

    while True:
        # 더 이상 read_json() 호출하지 않음 (초기 메시지 제외)

        # send initial_json
        send_json(sock, initial_json)
        
        # 이미지 데이터 수신
        img_bytes = read_image(sock)
        if img_bytes is None:
            break

        if len(img_bytes) < 100:
            continue

        # Flask 서버에 업로드
        try:
            resp = requests.post(FLASK_SERVER_URL, files={'file': ('render.png', img_bytes, 'image/png')})
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    main()

