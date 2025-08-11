import os
import numpy as np
import torch
from flask import Flask, jsonify
from argparse import ArgumentParser

app = Flask(__name__)

# -------------------- 입력 받기 --------------------
parser = ArgumentParser(description="NPZ Pose Frame Server")
parser.add_argument("--npz_path", type=str, required=True, help="Path to NPZ file containing pose data")
args = parser.parse_args()

# -------------------- 데이터 로드 --------------------
smpl_params = dict(np.load(args.npz_path))
thetas_all = smpl_params["poses"][..., :72]
transl_all = smpl_params["trans"] - smpl_params["trans"][0:1]
transl_all += (0, 0.15, 5)
betas = smpl_params["betas"]

frame_idx = 0

@app.route("/pose/<int:idx>", methods=["GET"])
def get_pose(idx):
    if idx >= len(transl_all):
        return jsonify({"end": True})
    pose_data = {
        "betas": betas.tolist(),
        "poses": thetas_all[idx:idx+1].tolist(),
        "trans": transl_all[idx:idx+1].tolist()
    }
    return jsonify(pose_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
