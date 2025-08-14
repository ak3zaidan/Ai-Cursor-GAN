import json
import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter1d

def load_data(filename="data.json"):
    with open(filename, "r") as f:
        return np.array(json.load(f)["data"], dtype=float)

def save_data(data_list, filename="augmented_data.json"):
    with open(filename, "w") as f:
        json.dump({"data": data_list}, f)

# --- Transformations ---
def scale(data, sx, sy):
    res = data.copy()
    res[:, 0] *= sx
    res[:, 1] *= sy
    return res

def translate(data, tx, ty):
    res = data.copy()
    res[:, 0] += tx
    res[:, 1] += ty
    return res

def rotate(data, angle_deg):
    a = math.radians(angle_deg)
    res = data.copy()
    x, y = res[:, 0], res[:, 1]
    res[:, 0] = x * math.cos(a) - y * math.sin(a)
    res[:, 1] = x * math.sin(a) + y * math.cos(a)
    return res

def add_noise(data, noise_level):
    res = data.copy()
    res[:, 0] += np.random.normal(0, noise_level, size=len(data))
    res[:, 1] += np.random.normal(0, noise_level, size=len(data))
    return res

def flip_horizontal(data):
    res = data.copy()
    res[:, 0] = -res[:, 0]
    return res

def flip_vertical(data):
    res = data.copy()
    res[:, 1] = -res[:, 1]
    return res

def change_speed(data, factor):
    res = data.copy()
    t0 = res[0, 2]
    res[:, 2] = t0 + (res[:, 2] - t0) / factor
    return res

def elastic_distortion(data, alpha=1.0, sigma=5.0):
    res = data.copy()
    dx = np.random.normal(0, alpha, len(data))
    dy = np.random.normal(0, alpha, len(data))
    dx = gaussian_filter1d(dx, sigma=sigma)
    dy = gaussian_filter1d(dy, sigma=sigma)
    res[:, 0] += dx
    res[:, 1] += dy
    return res

def sine_warp(data, amp=5, freq=0.1):
    res = data.copy()
    res[:, 0] += amp * np.sin(np.arange(len(data)) * freq)
    res[:, 1] += amp * np.cos(np.arange(len(data)) * freq)
    return res

def drift_over_time(data, drift_x=0.1, drift_y=0.1):
    res = data.copy()
    n = len(data)
    res[:, 0] += np.linspace(0, drift_x * n, n)
    res[:, 1] += np.linspace(0, drift_y * n, n)
    return res

def jitter_bursts(data, prob=0.1, max_jitter=10):
    res = data.copy()
    for i in range(len(data)):
        if random.random() < prob:
            res[i, 0] += random.uniform(-max_jitter, max_jitter)
            res[i, 1] += random.uniform(-max_jitter, max_jitter)
    return res

def variable_speed(data):
    res = data.copy()
    t = res[:, 2]
    factors = np.linspace(0.5, 1.5, len(data) - 1)  # matches np.diff length
    dt_scaled = np.diff(t) * factors
    new_t = np.concatenate(([t[0]], t[0] + np.cumsum(dt_scaled)))
    res[:, 2] = new_t
    return res

def drop_points(data, drop_prob=0.05):
    mask = np.random.rand(len(data)) > drop_prob
    return data[mask]

def insert_points(data, insert_prob=0.05):
    res = []
    for i in range(len(data)-1):
        res.append(data[i])
        if random.random() < insert_prob:
            midpoint = (data[i] + data[i+1]) / 2
            res.append(midpoint)
    res.append(data[-1])
    return np.array(res)

def reverse_path(data):
    res = data[::-1].copy()
    t0 = res[0, 2]
    res[:, 2] = t0 + (res[:, 2] - t0)
    return res

def hesitation_stops(data, stop_prob=0.05, stop_duration=50):
    res = []
    for point in data:
        res.append(point)
        if random.random() < stop_prob:
            pause_point = point.copy()
            pause_point[2] += stop_duration
            res.append(pause_point)
    return np.array(res)

# --- Augmentation Pipeline ---
if __name__ == "__main__":
    original = load_data()
    augmented_sets = [original.tolist()]

    transforms = [
        scale, translate, rotate, add_noise, flip_horizontal, flip_vertical,
        change_speed, elastic_distortion, sine_warp, drift_over_time,
        jitter_bursts, variable_speed, drop_points, insert_points,
        reverse_path, hesitation_stops
    ]

    for _ in range(300):  # generate 300 variations
        data = original.copy()
        for transform in random.sample(transforms, k=random.randint(3, 7)):
            kwargs = {}
            if transform == scale:
                kwargs = {"sx": random.uniform(0.7, 1.3), "sy": random.uniform(0.7, 1.3)}
            elif transform == translate:
                kwargs = {"tx": random.uniform(-100, 100), "ty": random.uniform(-100, 100)}
            elif transform == rotate:
                kwargs = {"angle_deg": random.uniform(-25, 25)}
            elif transform == add_noise:
                kwargs = {"noise_level": random.uniform(0.5, 5.0)}
            elif transform == change_speed:
                kwargs = {"factor": random.uniform(0.5, 1.5)}
            elif transform == elastic_distortion:
                kwargs = {"alpha": random.uniform(0.5, 3.0), "sigma": random.uniform(2, 8)}
            elif transform == sine_warp:
                kwargs = {"amp": random.uniform(2, 10), "freq": random.uniform(0.05, 0.2)}
            elif transform == drift_over_time:
                kwargs = {"drift_x": random.uniform(-0.2, 0.2), "drift_y": random.uniform(-0.2, 0.2)}
            elif transform == jitter_bursts:
                kwargs = {"prob": random.uniform(0.05, 0.2), "max_jitter": random.uniform(2, 15)}
            elif transform == drop_points:
                kwargs = {"drop_prob": random.uniform(0.01, 0.1)}
            elif transform == insert_points:
                kwargs = {"insert_prob": random.uniform(0.01, 0.1)}
            elif transform == hesitation_stops:
                kwargs = {"stop_prob": random.uniform(0.02, 0.1), "stop_duration": random.randint(30, 150)}

            data = transform(data, **kwargs)

        augmented_sets.append(data.tolist())

    save_data(augmented_sets)
    print(f"Saved {len(augmented_sets)} augmented datasets to augmented_data.json")
