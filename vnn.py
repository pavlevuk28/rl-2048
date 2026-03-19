import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import (
    Move,
    apply_move,
    add_new_tile,
    get_new_board,
    possible_next_boards,
)

GAMMA = 0.999


class VNN(nn.Module):
    def __init__(self, device, intermediate_size=256, conv_out_size=128):
        super(VNN, self).__init__()
        self.device = device

        self.my_transformation = nn.Parameter(
            torch.randn(4, intermediate_size, device=self.device)
        )

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
            512,
        )

        self.linear_2 = nn.Linear(512, 512)
        self.linear_3 = nn.Linear(512, 512)
        self.linear_4 = nn.Linear(512, 256)
        self.linear_5 = nn.Linear(256, 128)
        self.linear_6 = nn.Linear(128, 64)
        self.linear_7 = nn.Linear(64, 1)

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
        log_boards = np.zeros_like(boards, dtype="float32")
        mask = boards != 0

        log_boards[mask] = np.log2(boards[mask])

        return self.forward(torch.tensor(log_boards, device=self.device))

    def best_moves_and_implied_values(self, boards):
        with torch.no_grad():
            n = len(boards)
            reward_fn = [{m.value: 0 for m in list(Move)} for _ in range(n)]
            next_value_fn = [{m.value: 0 for m in list(Move)} for _ in range(n)]

            # Collect all (board_idx, move_name, next_board, prob) in one pass
            all_entries = []
            for board_idx, board in enumerate(boards):
                for move in list(Move):
                    new_board, _, is_changed = apply_move(board, move)
                    reward_fn[board_idx][move.value] = int(is_changed)
                    if is_changed:
                        for b, p in possible_next_boards(new_board):
                            all_entries.append(
                                {
                                    "board_idx": board_idx,
                                    "move": move.name,
                                    "board": b,
                                    "prob": p,
                                }
                            )

            # Chunked forward pass to avoid OOM on large batches
            CHUNK_SIZE = 1024
            if len(all_entries) > 0:
                all_values = []
                for chunk_start in range(0, len(all_entries), CHUNK_SIZE):
                    chunk = all_entries[chunk_start : chunk_start + CHUNK_SIZE]
                    chunk_values = self.forward_from_int_boards([e["board"] for e in chunk]).flatten().tolist()
                    all_values.extend(chunk_values)
                values = all_values

                df = pd.DataFrame(
                    {
                        "board_idx": [e["board_idx"] for e in all_entries],
                        "move": [e["move"] for e in all_entries],
                        "values": values,
                        "probs": [e["prob"] for e in all_entries],
                    }
                )
                df["value_x_prob"] = df["values"] * df["probs"]

                for (board_idx, move_name), group in df.groupby(["board_idx", "move"]):
                    next_value_fn[board_idx][move_name] = group["value_x_prob"].sum()

            best_moves_and_implied_values = []
            for board_idx in range(n):
                moves_and_implied_values = [
                    {
                        "move": move,
                        "implied_value": reward_fn[board_idx][move.value]
                        + GAMMA * next_value_fn[board_idx][move.value],
                    }
                    for move in list(Move)
                ]
                best_move_and_implied_value = max(
                    moves_and_implied_values,
                    key=lambda x: x["implied_value"],
                )
                best_moves_and_implied_values.append(best_move_and_implied_value)

            return best_moves_and_implied_values

    def play_games_and_get_trajectories(self, num_games, eps):
        boards = [get_new_board() for _ in range(num_games)]
        trajectories = [[] for _ in range(num_games)]
        alive = [True] * num_games

        while any(alive):
            alive_indices = [i for i, a in enumerate(alive) if a]
            alive_boards = [boards[i] for i in alive_indices]

            best_moves_and_implied_values = self.best_moves_and_implied_values(
                alive_boards
            )

            for game_idx, best_move_and_implied_value in zip(
                alive_indices, best_moves_and_implied_values
            ):
                board = boards[game_idx]

                new_board, _, changed = apply_move(
                    board,
                    (
                        random.choice(list(Move))
                        if np.random.rand() < eps
                        else best_move_and_implied_value["move"]
                    ),
                )

                if not changed:
                    alive[game_idx] = False
                else:
                    new_board_with_new_tile = add_new_tile(new_board)

                    boards[game_idx] = new_board_with_new_tile
                    trajectories[game_idx].append(new_board_with_new_tile)

        return trajectories
