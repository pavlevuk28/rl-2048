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

EPOCHS = 40
NEW_GAMES_PER_EPOCH = 10
TIMES_TO_BACKPROP_PER_EPOCH = 20
MINI_BATCH_SIZE = 1024
EPS_START = 0.9
EPS_END = 0.0001
EPS_DECAY = 4
LR = 3e-2


@app.function(
    gpu="A100",
    image=image,
    timeout=3600,
)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vnn = VNN(device=device).to(device)
    vnn.share_memory()
    optimizer = optim.AdamW(vnn.parameters(), lr=LR, amsgrad=True)

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

        print(
            "Average lengths of games",
            np.array([len(traj) for traj in trajectories_from_this_epoch]).mean(),
        )

        boards_to_train_on = [
            board
            for game_trajectory in trajectories_from_this_epoch
            for board in game_trajectory
        ]

        if len(boards_to_train_on) > 0:
            criterion = nn.SmoothL1Loss()
            for _ in range(TIMES_TO_BACKPROP_PER_EPOCH):
                indices = np.random.choice(
                    len(boards_to_train_on),
                    size=min(MINI_BATCH_SIZE, len(boards_to_train_on)),
                    replace=False,
                )
                mini_batch = [boards_to_train_on[i] for i in indices]

                target = [
                    best_move_and_implied_value["implied_value"]
                    for best_move_and_implied_value in vnn.best_moves_and_implied_values(
                        mini_batch
                    )
                ]

                loss = criterion(
                    vnn.forward_from_int_boards(mini_batch).reshape(-1),
                    torch.tensor(target, device=device),
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_value_(vnn.parameters(), 100)
                optimizer.step()

    return vnn
