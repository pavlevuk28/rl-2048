import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

from engine import (
    Move,
    Game_state,
    apply_move,
    add_new_tile,
    get_new_board,
    possible_next_boards,
)

GAMMA = 0.999


class VNN(nn.Module):
    def __init__(self, intermediate_size=100, conv_out_size=10):
        super(VNN, self).__init__()

        self.my_transformation = nn.Parameter(torch.randn(4, intermediate_size))

        self.horizontal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_size,
            kernel_size=(1, 2),
            stride=1,
            padding=0,
        )

        self.vertical_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_size,
            kernel_size=(2, 1),
            stride=1,
            padding=0,
        )

        self.linear_1 = nn.Linear(
            2 * conv_out_size * 4 * 3 + intermediate_size**2,
            256,
        )

        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 256)
        self.linear_4 = nn.Linear(256, 128)
        self.linear_5 = nn.Linear(128, 64)
        self.linear_6 = nn.Linear(64, 16)
        self.linear_7 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        mmult_result = self.my_transformation.T @ x @ self.my_transformation

        horizontal_conv_result = self.horizontal_conv(x)
        vertical_conv_result = self.vertical_conv(x)

        x = torch.cat(
            (
                mmult_result.reshape(x.size(0), -1),
                horizontal_conv_result.reshape(x.size(0), -1),
                vertical_conv_result.reshape(x.size(0), -1),
            ),
            dim=1,
        )

        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = F.relu(self.linear_5(x))
        x = F.relu(self.linear_6(x))

        return self.linear_7(x)

    def forward_from_int_boards(self, boards):
        boards = np.array([b for b in boards], dtype="float32")
        log_board = np.zeros_like(boards, dtype="float32")
        mask = boards != 0

        log_board[mask] = np.log2(boards[mask])

        return self.forward(torch.tensor(log_board))

    def best_move_and_implied_value(self, board):
        with torch.no_grad():
            next_value_fn_dict = dict()
            reward_fn_dict = {m: 0 for m in list(Move)}

            move_board_probability = []

            for move in list(Move):
                new_board, score_change, changed = apply_move(board, move)

                reward_fn_dict[move.value] = int(changed)
                # reward_fn_dict[move.value] = (
                #     np.log2(score_change) ** 2 if score_change > 0 else 0
                # )

                if not changed:
                    next_value_fn_dict[move.value] = 0
                else:
                    move_board_probability = move_board_probability + (
                        [(move, b, p) for b, p in possible_next_boards(new_board)]
                    )

            moves = [m.name for m, _, _ in move_board_probability]
            boards = [b for _, b, _ in move_board_probability]
            values = (
                self.forward_from_int_boards(boards).flatten().tolist()
                if len(boards) > 0
                else []
            )
            probabilities = [p for _, _, p in move_board_probability]

            df = pd.DataFrame({"move": moves, "values": values, "probs": probabilities})
            df["move"] = df["move"].astype(str)
            df["value_x_prob"] = df.eval("values * probs")

            next_value_fn_dict = next_value_fn_dict | dict(
                df.groupby("move")["value_x_prob"].sum()
            )

            value_fn_dict = {
                m.value: reward_fn_dict[m.value] + GAMMA * next_value_fn_dict[m.value]
                for m in list(Move)
            }

            best_move = max(
                list(Move),
                key=lambda m: reward_fn_dict[m.value]
                + GAMMA * next_value_fn_dict[m.value],
            )

            return best_move, value_fn_dict[best_move.value]

    def play_game_and_get_trajectory(self, eps):
        alive = True
        board = get_new_board()
        trajectory = []

        while alive:
            best_move, implied_value_from_best_move = self.best_move_and_implied_value(
                board
            )
            new_board, reward, changed = apply_move(board, best_move)

            if np.random.rand() < eps:
                rand_move = random.choice(list(Move))
                new_board, _, changed = apply_move(board, rand_move)

            trajectory.append(
                Game_state(
                    board=board,
                    new_board=new_board,
                    best_move=best_move,
                    implied_value_from_best_move=implied_value_from_best_move,
                )
            )

            board = new_board

            if not changed:
                alive = False

                trajectory.append(
                    Game_state(
                        board=board,
                        new_board=None,
                        best_move=None,
                        implied_value_from_best_move=0,
                    )
                )
            else:
                board = add_new_tile(board)

        return trajectory
