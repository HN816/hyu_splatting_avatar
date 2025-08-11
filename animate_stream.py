import requests
import torch
import os
import numpy as np
from pathlib import Path

from model.splatting_avatar_model import SplattingAvatarModel
from dataset.dataset_helper import make_frameset_data
from scene.dataset_readers import make_scene_camera
from gaussian_renderer import network_gui
from model import libcore

SERVER_URL = "http://127.0.0.1:9000/pose"

def get_pose_from_server(frame_idx):
    res = requests.get(f"{SERVER_URL}/{frame_idx}").json()
    if res.get("end"):
        return None

    global_orient = torch.tensor(res["global_orient"]).float()  # shape: [1, 3]
    body_pose = torch.tensor(res["body_pose"]).float()          # shape: [1, 69]
    transl = torch.tensor(res["transl"]).float()                # shape: [1, 3]

    # shapes must be:
    # global_orient: [1, 3]
    # body_pose: [1, 69]
    # transl: [1, 3]

    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
    }

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='SplattingAvatar Server Streaming')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=True)
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], 
                        required=True, help='path to config file')
    parser.add_argument('--pc_dir', type=str, default=None)
    args, extras = parser.parse_known_args()

    # config 로드
    config = libcore.load_from_config(args.configs, cli_args=extras)
    config.dataset.dat_dir = args.dat_dir
    frameset_train = make_frameset_data(config.dataset, split='train')

    smpl_model = frameset_train.smpl_model
    cam = frameset_train.cam
    empty_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)
    viewpoint_cam = make_scene_camera(0, cam, empty_img)
    mesh_py3d = frameset_train.mesh_py3d

    betas = frameset_train.smpl_params['betas']

    pipe = config.pipe
    gs_model = SplattingAvatarModel(config.model, verbose=True)
    ply_fn = os.path.join(args.pc_dir, 'point_cloud.ply')
    gs_model.load_ply(ply_fn)
    embed_fn = os.path.join(args.pc_dir, 'embedding.json')
    gs_model.load_from_embedding(embed_fn)

    if args.ip != 'none':
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None

    frame_idx = 0
    while True:
        pose_params = get_pose_from_server(frame_idx)
        if pose_params is None:
            break

        pose_params["betas"] = betas
        out = smpl_model(**pose_params)
        frame_mesh = mesh_py3d.update_padded(out['vertices'])
        mesh_info = {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }

        gs_model.update_to_posed_mesh(mesh_info)
        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background='white')
        image = render_pkg['render']

        if verify is not None:
            network_gui.send_image_to_network(image, verify)

        frame_idx += 1

    print("[done]")
