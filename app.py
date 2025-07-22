import os
import threading
from argparse import ArgumentParser
from model.splatting_avatar_model import SplattingAvatarModel
from dataset.dataset_helper import make_frameset_data
from model import libcore
from gaussian_renderer import network_gui
from run_testing_live import run_testing_live  # run_testing_live는 저장 안 하고 send_image_to_network 호출하는 평가 함수

if __name__ == '__main__':
    parser = ArgumentParser(description='SplattingAvatar Evaluation Live Render')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--tcp_port', type=int, default=6009)
    parser.add_argument('--web_port', type=int, default=6010)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], required=True)
    parser.add_argument('--pc_dir', type=str, default=None)
    args, extras = parser.parse_known_args()

    config = libcore.load_from_config(args.configs, cli_args=extras)
    config.dataset.dat_dir = args.dat_dir
    frameset_test = make_frameset_data(config.dataset, split='test')

    gs_model = SplattingAvatarModel(config.model, verbose=True)
    ply_fn = os.path.join(args.pc_dir, 'point_cloud.ply')
    gs_model.load_ply(ply_fn)
    embed_fn = os.path.join(args.pc_dir, 'embedding.json')
    gs_model.load_from_embedding(embed_fn)

    print(f"[DEBUG] ip: {args.ip}, port: {args.tcp_port}, type(port): {type(args.tcp_port)}")
    network_gui.init(args.ip, args.tcp_port)

    # 웹서버 별도 스레드 실행 (TCP 포트 +1)
    threading.Thread(target=network_gui.start_web_server, args=(args.ip, args.web_port), daemon=True).start()

    # 평가 실행 (이미지 저장 없이 네트워크 전송)
    run_testing_live(config.pipe, frameset_test, gs_model, args.pc_dir, verify=None)

    print('[done]')
