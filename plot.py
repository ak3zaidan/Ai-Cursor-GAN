import json
import numpy as np
import matplotlib.pyplot as plt

# Load the data from file
with open("data.json", "r") as f:
    data = json.load(f)["data"]

# Convert to NumPy arrays for easier math
data = np.array(data, dtype=float)  # shape: (n, 3)
x = data[:, 0]
y = data[:, 1]
t = data[:, 2] / 1000.0  # convert ms → seconds

# Position magnitude (distance from origin)
position = np.sqrt(x**2 + y**2)

# Compute velocities (magnitude of dx/dt and dy/dt)
dt = np.diff(t)
dx = np.diff(x)
dy = np.diff(y)
velocity = np.sqrt(dx**2 + dy**2) / dt  # pixels/sec

# Compute accelerations
dv = np.diff(velocity)
acceleration = dv / dt[1:]  # pixels/sec^2

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

# 1. Position vs Time
axs[0].plot(t, position, marker='o', markersize=2)
axs[0].set_ylabel("Position (pixels)")
axs[0].set_title("Position vs Time")

# 2. Velocity profile
axs[1].plot(t[1:], velocity, marker='o', markersize=2, color='orange')
axs[1].set_ylabel("Velocity (pixels/sec)")
axs[1].set_title("Velocity Profile")

# 3. Acceleration profile
axs[2].plot(t[2:], acceleration, marker='o', markersize=2, color='red')
axs[2].set_ylabel("Acceleration (pixels/sec²)")
axs[2].set_xlabel("Time (sec)")
axs[2].set_title("Acceleration Profile")

plt.tight_layout()
plt.show()
