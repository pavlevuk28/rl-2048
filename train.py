from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from engine import *
from vnn import *
import modal

app = modal.App("rl-2048")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_python_source("vnn", "engine")
)

EPOCHS = 100
BATCH_SIZE = 30_000
NEW_GAMES_PER_EPOCH = 100
EPS_START = 0.9
EPS_END = 0.0001
EPS_DECAY = 10
LR = 3e-4


@app.function(
    gpu="T4",
    image=image,
    timeout=3600,
)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vnn = VNN(device=device).to(device)
    vnn.share_memory()
    optimizer = optim.AdamW(vnn.parameters(), lr=LR, amsgrad=True)

    trajectories_by_epoch = dict()

    for epoch in range(EPOCHS):
        print("Epoch:", epoch)

        trajectories_from_this_epoch = []

        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
            -1.0 * epoch / EPS_DECAY
        )
        print("eps", eps_threshold)
        trajectories_from_this_epoch = vnn.play_games_and_get_trajectories(
            num_games=NEW_GAMES_PER_EPOCH, eps=eps_threshold
        )

        trajectories_by_epoch[epoch] = trajectories_from_this_epoch

        print(
            "Average lengths of games",
            np.array([len(traj) for traj in trajectories_from_this_epoch]).mean(),
        )

        states_to_train_on = [
            state
            for trajectories_from_game in trajectories_from_this_epoch
            for state in trajectories_from_game
        ]
        boards = [state.board for state in states_to_train_on]

        target = [state.implied_value_from_best_move for state in states_to_train_on]

        criterion = nn.SmoothL1Loss()
        loss = criterion(
            vnn.forward_from_int_boards(boards).reshape(-1),
            torch.tensor(target, device=device),
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(vnn.parameters(), 100)
        optimizer.step()

    return vnn
