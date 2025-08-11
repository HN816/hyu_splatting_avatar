from flask import Flask, request, render_template
from flask_socketio import SocketIO
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

latest_image_b64 = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webgl')
def webgl():
    return render_template('webgl.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global latest_image_b64
    file = request.files.get('file')
    if file:
        image_bytes = file.read()
        latest_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        print(f"[Flask] 이미지 수신됨, 크기: {len(image_bytes)} bytes")
        return "OK", 200
    return "No file", 400

def send_image_loop():
    global latest_image_b64
    while True:
        if latest_image_b64 is not None:
            socketio.emit('new_frame', {'image': latest_image_b64})
        socketio.sleep(0.03) 

@socketio.on('connect')
def on_connect():
    print("[Flask] 클라이언트 연결됨")


if __name__ == '__main__':
    socketio.start_background_task(send_image_loop)
    socketio.run(app, host='0.0.0.0', port=8000)
