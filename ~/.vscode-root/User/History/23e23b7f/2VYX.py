import threading
import time
from io import BytesIO

from PIL import Image
from flask import Flask
from flask_socketio import SocketIO

import os
import torch
import gc

# 환경변수 설정은 실행 초기에 한 번만!
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Gaussian Splatting 관련 모듈
from model.splatting_avatar_model import SplattingAvatarModel
from model import libcore
from dataset.dataset_helper import make_frameset_data

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

gs_model = None
viewpoint_cam = None
pipe = None

frame_lock = threading.Lock()
current_frame_jpeg = None

def init_model_and_camera(config_path, datadir, pc_dir, flame_params=None):
    global gs_model, viewpoint_cam, pipe

    config = libcore.load_from_config([config_path], cli_args=[])
    config.dataset.dat_dir = datadir
    if flame_params:
        config.dataset.flame_param_path = flame_params

    frameset_test = make_frameset_data(config.dataset, split='test')

    gs_model = SplattingAvatarModel(config.model, verbose=True)
    gs_model.eval()

    ply_fn = os.path.join(pc_dir, "point_cloud.ply")
    embed_fn = os.path.join(pc_dir, "embedding.json")
    gs_model.load_ply(ply_fn)
    gs_model.load_from_embedding(embed_fn)

    pipe = config.pipe

    batch = frameset_test[0]
    viewpoint_cam = batch['scene_cameras'][0].cuda()

def render_frame():
    global gs_model, viewpoint_cam, pipe
    with torch.no_grad():
        # 렌더링 전 메모리 여유를 위해 간헐적으로 호출하는 것을 권장
        # torch.cuda.empty_cache() 너무 자주 호출시 오히려 성능 저하 가능
        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background='white', scaling_modifer=0.028)

    # GPU 텐서를 CPU numpy로 변환: 최소화 필요
    img_tensor = render_pkg["render"]  # (C,H,W), float32 in [0,1]
    img_tensor = img_tensor[:3]  # RGBA -> RGB (필요시)
    img_np = (img_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

    # PIL 변환과 리사이즈는 여기서 한번만 수행
    pil_img = Image.fromarray(img_np)
    pil_img = pil_img.resize((640, 480), Image.LANCZOS)

    buff = BytesIO()
    pil_img.save(buff, format='JPEG', quality=85)  # quality 조절 가능
    jpeg_bytes = buff.getvalue()

    # 필요에 따라 GC 및 CUDA 캐시 정리 (너무 자주 호출 주의)
    gc.collect()

    return jpeg_bytes

def frame_producer():
    global current_frame_jpeg, frame_lock
    while True:
        jpeg_bytes = render_frame()
        with frame_lock:
            current_frame_jpeg = jpeg_bytes
        time.sleep(1 / 30)  # 30 FPS 목표

@socketio.on('connect')
def on_connect():
    print('Client connected')

def send_frame():
    global current_frame_jpeg, frame_lock
    while True:
        with frame_lock:
            if current_frame_jpeg is not None:
                socketio.emit('frame', current_frame_jpeg)
        socketio.sleep(1 / 30)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head><title>Gaussian Splatting Realtime Streaming</title></head>
    <body>
        <h1>Realtime Render Stream</h1>
        <img id="stream" width="640" height="480" />
        <p id="url-display"></p>
        <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
        <script>
            const socket = io();
            let oldUrl = null;
            socket.on('frame', function(data) {
                if (oldUrl) {
                    URL.revokeObjectURL(oldUrl);
                }
                let arrayBufferView = new Uint8Array(data);
                let blob = new Blob([arrayBufferView], {type: "image/jpeg"});
                let urlCreator = window.URL || window.webkitURL;
                let imageUrl = urlCreator.createObjectURL(blob);
                document.getElementById('stream').src = imageUrl;
                oldUrl = imageUrl;
                document.getElementById('url-display').textContent = imageUrl;
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dat_dir', required=True)
    parser.add_argument('--pc_dir', required=True)
    parser.add_argument('--flame_params', default=None)
    args = parser.parse_args()

    torch.cuda.empty_cache()

    print("Initializing model and camera...")
    init_model_and_camera(args.config, args.dat_dir, args.pc_dir, args.flame_params)

    print("Starting frame producer thread...")
    threading.Thread(target=frame_producer, daemon=True).start()

    print("Starting background frame sender...")
    socketio.start_background_task(send_frame)

    print("Starting Flask-SocketIO server on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
