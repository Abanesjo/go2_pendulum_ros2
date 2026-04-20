#!/usr/bin/env python3
"""Plot pendulum joint positions and velocities from a /joint_states CSV log."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FILENAME = 'center_joint1.csv'


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
    plt.show()


if __name__ == '__main__':
    main()
