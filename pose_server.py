import os
import numpy as np
import torch
from flask import Flask, jsonify
from argparse import ArgumentParser

app = Flask(__name__)

class AnimateDataset(torch.utils.data.Dataset):
    def __init__(self, pose_sequence_path):
        smpl_params = dict(np.load(pose_sequence_path))
        thetas = smpl_params["poses"][..., :72]
        transl = smpl_params["trans"] - smpl_params["trans"][0:1]
        transl += (0, 0.15, 5)

        self.thetas = torch.tensor(thetas).float()
        self.transl = torch.tensor(transl).float()

    def __len__(self):
        return len(self.transl)

    def __getitem__(self, idx):
        return {
            "global_orient": self.thetas[idx:idx+1, :3].tolist(),
            "body_pose": self.thetas[idx:idx+1, 3:].tolist(),
            "transl": self.transl[idx:idx+1].tolist(),
        }

# argparse로 NPZ 경로 받기
parser = ArgumentParser(description="NPZ Pose Frame Server")
parser.add_argument("--npz_path", type=str, required=True, help="Path to NPZ file containing pose data")
args = parser.parse_args()

# AnimateDataset 인스턴스 생성
anim_data = AnimateDataset(args.npz_path)

@app.route("/pose/<int:idx>", methods=["GET"])
def get_pose(idx):
    if idx < 0 or idx >= len(anim_data):
        return jsonify({"end": True})

    pose = anim_data[idx]
    return jsonify(pose)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
