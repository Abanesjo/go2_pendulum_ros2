#!/usr/bin/env python3
"""Plot pendulum state and LowCmd joint targets from a CSV log."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FILENAME = 'data.csv'
LOWCMD_TARGET_COLUMNS = [
    'FR_hip_joint_lowcmd_target',
    'FR_thigh_joint_lowcmd_target',
    'FR_calf_joint_lowcmd_target',
    'FL_hip_joint_lowcmd_target',
    'FL_thigh_joint_lowcmd_target',
    'FL_calf_joint_lowcmd_target',
    'RR_hip_joint_lowcmd_target',
    'RR_thigh_joint_lowcmd_target',
    'RR_calf_joint_lowcmd_target',
    'RL_hip_joint_lowcmd_target',
    'RL_thigh_joint_lowcmd_target',
    'RL_calf_joint_lowcmd_target',
]


def main():
    csv_path = Path(__file__).parent / FILENAME
    data = np.genfromtxt(csv_path, delimiter=',', names=True)

    t = data['timestamp'] - data['timestamp'][0]

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 6))

    axes[0, 0].plot(t, np.rad2deg(data['pendulum_joint1_pos']))
    axes[0, 0].set_title('pendulum_joint1')
    axes[0, 0].set_ylabel('position (deg)')
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, np.rad2deg(data['pendulum_joint2_pos']))
    axes[0, 1].set_title('pendulum_joint2')
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, np.rad2deg(data['pendulum_joint1_vel']))
    axes[1, 0].set_ylabel('velocity (deg/s)')
    axes[1, 0].set_xlabel('time (s)')
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, np.rad2deg(data['pendulum_joint2_vel']))
    axes[1, 1].set_xlabel('time (s)')
    axes[1, 1].grid(True)

    fig.suptitle(f'Pendulum joints — {FILENAME}')
    plt.tight_layout()

    target_columns = [c for c in LOWCMD_TARGET_COLUMNS if c in data.dtype.names]
    if target_columns:
        fig_targets, ax_targets = plt.subplots(figsize=(12, 6))
        for col in target_columns:
            label = col.removesuffix('_lowcmd_target')
            ax_targets.plot(t, data[col], label=label)
        ax_targets.set_title(f'LowCmd joint targets — {FILENAME}')
        ax_targets.set_xlabel('time (s)')
        ax_targets.set_ylabel('target position (rad)')
        ax_targets.grid(True)
        ax_targets.legend(ncol=3, fontsize='small')
        plt.tight_layout()
    else:
        print('No *_lowcmd_target columns found; skipping LowCmd target plot.')

    plt.show()


if __name__ == '__main__':
    main()
