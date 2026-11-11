"""Numerical scene helpers extracted from SceneInitializationCrossing.py; plotting excluded."""

import copy
import json
import os
import random
import numpy as np
from ._scene_parameters import swarm_type_get, sensitivity_types_get
from ._scene_swarm import generate_large_swarm
from scipy.interpolate import CubicSpline
from ._scene_parameters import generate_fixed_parameters_cross
L = 40.0


class Node:
    _id_counter = 0

    def __init__(self, type, pos=(0, 0), velocity=(0, 0), task=None, ability=1, group_index=None):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.type = type
        self.pos = np.array(pos)
        self.velocity = np.array(velocity)
        self.task = task
        self.parameters = generate_fixed_parameters_cross()
        self.ability = ability
        self.group_index = group_index
        self.initialize()

    def initialize(self):
        if self.type not in ('relay', 'sensing', 'action', 'allround') and 'environment' not in self.type:
            raise ValueError(f"Invalid type: {self.type}. Type must be 'relay', 'sensing', 'action', or contain 'environment'.")
        if 'environment' in self.type:
            self.velocity = self.velocity * self.parameters[f'velocity_shill']
        else:
            self.acceleration_max = self.parameters[f'initial_acceleration_max'][f'{self.type}']
            self.velocity_max = self.parameters[f'initial_velocity_max'][f'{self.type}']
            theta = 0
            self.velocity = 5 * np.array([np.cos(theta), np.sin(theta)])
            if self.type == 'relay':
                self.r_c = self.parameters[f'initial_range_communication'][f'{self.type}']
            elif self.type == 'sensing':
                self.r_s = self.parameters[f'initial_range_sensing'][f'{self.type}']
                self.r_c = self.parameters[f'initial_range_communication'][f'{self.type}']
            elif self.type == 'action':
                self.r_a = self.parameters[f'initial_range_action'][f'{self.type}']
                self.r_c = self.parameters[f'initial_range_communication'][f'{self.type}']
            elif self.type == 'allround':
                self.r_a = self.parameters[f'initial_range_action'][f'{self.type}']
                self.r_s = self.parameters[f'initial_range_sensing'][f'{self.type}']
                self.r_c = self.parameters[f'initial_range_communication'][f'{self.type}']

    def to_dict(self):
        """将节点信息转换为字典形式，以便保存到JSON文件中"""

        def format_float(value):
            return round(float(value), 6) if value is not None else None
        data = {'id': self.id, 'type': self.type, 'pos': [format_float(v) for v in self.pos], 'velocity': [format_float(v) for v in self.velocity], 'velocity_max': format_float(getattr(self, 'velocity_max', None)), 'acceleration_max': format_float(getattr(self, 'acceleration_max', None)), 'r_s': format_float(getattr(self, 'r_s', None)), 'r_c': format_float(getattr(self, 'r_c', None)), 'r_a': format_float(getattr(self, 'r_a', None)), 'task': self.task}
        if self.group_index is not None:
            data['group_index'] = int(self.group_index)
        return data

def reset_random_seed(seed=None):
    """Reset numpy/python random seeds when deterministic environments are needed."""
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)

def clone_nodes_with_ranges(nodes, relay_comm_range, action_range, sensing_range=None):
    """Deep copy nodes and override relay/sensing/action ranges for sensitivity tests."""
    cloned = copy.deepcopy(nodes)
    for node in cloned:
        if node.type == 'relay':
            node.r_c = relay_comm_range
        elif sensing_range is not None and node.type == 'sensing':
            node.r_s = sensing_range
        elif node.type == 'action':
            node.r_a = action_range
    return cloned

def _randomize_channel_control_points(base_upper, base_lower, center_min=40.0, center_max=200.0):
    num_points = len(base_upper)
    upper = np.array(base_upper, dtype=float, copy=True)
    lower = np.array(base_lower, dtype=float, copy=True)
    if num_points <= 2:
        return (upper, lower)
    y_diffs = upper[:, 1] - lower[:, 1]
    for idx in range(1, num_points - 1):
        center_y = np.random.uniform(center_min, center_max)
        half_diff = y_diffs[idx] / 2.0
        upper[idx, 1] = center_y + half_diff
        lower[idx, 1] = center_y - half_diff
    return (upper, lower)

def _circle_profile_with_derivatives(x, circle_center, radius, upper=True):
    dx = x - circle_center[0]
    inside = np.maximum(radius * radius - dx * dx, 0.0)
    delta = np.sqrt(inside)
    if upper:
        y = circle_center[1] + delta
        sign = 1.0
    else:
        y = circle_center[1] - delta
        sign = -1.0
    denom = np.sqrt(np.maximum(inside, 1e-12))
    dy = sign * (-dx / denom)
    d2y = sign * (-(radius * radius) / denom ** 3)
    return (y, dy, d2y)

def _quintic_hermite_interpolate(x, x0, x1, y0, dy0, d2y0, y1, dy1, d2y1):
    h = x1 - x0
    t = (x - x0) / h
    a0 = y0
    a1 = dy0 * h
    a2 = 0.5 * d2y0 * h * h
    b0 = y1 - (a0 + a1 + a2)
    b1 = dy1 * h - (a1 + 2.0 * a2)
    b2 = d2y1 * h * h - 2.0 * a2
    mat = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]], dtype=float)
    (a3, a4, a5) = np.linalg.solve(mat, np.array([b0, b1, b2], dtype=float))
    return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5

def _resample_by_arclength(x, y, step):
    if len(x) < 2:
        return (x, y)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]
    if total <= 0.0:
        return (x[:1], y[:1])
    targets = np.arange(0.0, total, step)
    if targets.size == 0 or total - targets[-1] > 1e-09:
        targets = np.append(targets, total)
    x_new = np.interp(targets, s, x)
    y_new = np.interp(targets, s, y)
    return (x_new, y_new)

def generate_channel():
    base_upper = np.array(generate_fixed_parameters_cross()['control_points_upper'], dtype=float)
    base_lower = np.array(generate_fixed_parameters_cross()['control_points_lower'], dtype=float)
    (control_points_upper, control_points_lower) = _randomize_channel_control_points(base_upper, base_lower, center_min=40.0, center_max=200.0)
    A = control_points_upper[0]
    B = control_points_lower[0]
    circle_center_y = (A[1] + B[1]) / 2
    circle_center = np.array([-circle_center_y, circle_center_y])
    radius = circle_center_y * np.sqrt(2)
    x0 = -L
    x1 = L
    angles_deg = np.arange(180.0, 44.0, -3.0)
    angles = np.deg2rad(angles_deg)
    x_left = circle_center[0] + radius * np.cos(angles)
    y_upper_left = circle_center[1] + radius * np.sin(angles)
    left_mask = x_left < x0 - 1e-09
    x_left = x_left[left_mask]
    y_upper_left = y_upper_left[left_mask]
    y_lower_left = circle_center[1] - np.sqrt(np.clip(radius ** 2 - (x_left - circle_center[0]) ** 2, 0.0, None))
    if x_left.size > 1:
        order = np.argsort(x_left)
        x_left = x_left[order]
        y_upper_left = y_upper_left[order]
        y_lower_left = y_lower_left[order]
    x_step_mid = 4.0
    x_step_right = 4.0
    x_max = max(np.max(control_points_upper[:, 0]), np.max(control_points_lower[:, 0]))
    x_mid = np.arange(x0, x1 + x_step_mid, x_step_mid)
    x_tail = np.arange(x1 + x_step_right, x_max + x_step_right, x_step_right)
    x_right = np.concatenate([x_mid, x_tail]) if x_tail.size else x_mid
    if x_right.size == 0 or x_right[-1] < x_max - 1e-09:
        x_right = np.append(x_right, x_max)
    spline_upper = CubicSpline(control_points_upper[:, 0], control_points_upper[:, 1])
    spline_lower = CubicSpline(control_points_lower[:, 0], control_points_lower[:, 1])
    y_upper_right = spline_upper(x_right)
    y_lower_right = spline_lower(x_right)
    (y0_u, dy0_u, d2y0_u) = _circle_profile_with_derivatives(x0, circle_center, radius, upper=True)
    (y0_l, dy0_l, d2y0_l) = _circle_profile_with_derivatives(x0, circle_center, radius, upper=False)
    y1_u = float(spline_upper(x1))
    dy1_u = float(spline_upper.derivative(1)(x1))
    d2y1_u = float(spline_upper.derivative(2)(x1))
    y1_l = float(spline_lower(x1))
    dy1_l = float(spline_lower.derivative(1)(x1))
    d2y1_l = float(spline_lower.derivative(2)(x1))
    mask_right = (x_right >= x0) & (x_right <= x1)
    if np.any(mask_right):
        y_upper_right[mask_right] = _quintic_hermite_interpolate(x_right[mask_right], x0, x1, y0_u, dy0_u, d2y0_u, y1_u, dy1_u, d2y1_u)
        y_lower_right[mask_right] = _quintic_hermite_interpolate(x_right[mask_right], x0, x1, y0_l, dy0_l, d2y0_l, y1_l, dy1_l, d2y1_l)
    y_upper = np.concatenate([y_upper_left, y_upper_right])
    y_lower = np.concatenate([y_lower_left, y_lower_right])
    x = np.concatenate([x_left, x_right])
    mask_all = (x >= x0) & (x <= x1)
    if np.any(mask_all):
        y_upper[mask_all] = _quintic_hermite_interpolate(x[mask_all], x0, x1, y0_u, dy0_u, d2y0_u, y1_u, dy1_u, d2y1_u)
        y_lower[mask_all] = _quintic_hermite_interpolate(x[mask_all], x0, x1, y0_l, dy0_l, d2y0_l, y1_l, dy1_l, d2y1_l)
    return (x, y_upper, y_lower, circle_center, radius, x_right, y_upper_right, y_lower_right)

def generate_obstacles(channel_x, channel_y_upper, channel_y_lower, num_obstacles=10):
    obstacles = []
    channel_x = np.asarray(channel_x, dtype=float)
    positive_x = channel_x[channel_x > 0.0]
    if positive_x.size == 0:
        raise ValueError('No positive x range available for obstacle generation.')
    x_min = float(np.min(positive_x))
    x_max = float(np.max(positive_x))
    while len(obstacles) < num_obstacles:
        obstacle_x = np.random.uniform(x_min, x_max)
        y_min = np.interp(obstacle_x, channel_x, channel_y_lower) + 10
        y_max = np.interp(obstacle_x, channel_x, channel_y_upper) - 10
        obstacle_y = np.random.uniform(y_min, y_max)
        if (obstacle_x, obstacle_y) not in obstacles:
            obstacles.append((obstacle_x, obstacle_y))
    return obstacles

def generate_obstacle_nodes(obstacles):
    obstacle_nodes = []
    for (obs_x, obs_y) in obstacles:
        velocity = np.array([1.0, 0.0])
        obstacle_nodes.append(Node('environmentObstacle', pos=(obs_x, obs_y), velocity=velocity))
    return obstacle_nodes

def calculate_boundary_velocity(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    norm = np.sqrt(dx ** 2 + dy ** 2)
    return np.array([dx / norm, dy / norm])

def generate_boundary_nodes(channel_x, channel_y_upper, channel_y_lower):
    boundary_nodes = []
    for (i, (x, y_upper, y_lower)) in enumerate(zip(channel_x, channel_y_upper, channel_y_lower)):
        if i < len(channel_x) - 1:
            (x1, y1) = (channel_x[i], channel_y_upper[i])
            (x2, y2) = (channel_x[i + 1], channel_y_upper[i + 1])
            upper_velocity = calculate_boundary_velocity(x1, y1, x2, y2)
            boundary_nodes.append(Node('environmentBoundary', pos=(x, y_upper), velocity=upper_velocity))
            (x1, y1) = (channel_x[i], channel_y_lower[i])
            (x2, y2) = (channel_x[i + 1], channel_y_lower[i + 1])
            lower_velocity = calculate_boundary_velocity(x1, y1, x2, y2)
            boundary_nodes.append(Node('environmentBoundary', pos=(x, y_lower), velocity=lower_velocity))
    return boundary_nodes

def generate_circular_boundary_nodes(circle_center, radius, num_points=100):
    boundary_nodes = []
    start_angle = np.deg2rad(45)
    end_angle = np.deg2rad(315)
    for angle in np.linspace(start_angle, end_angle, num_points):
        node_x = circle_center[0] + radius * np.cos(angle)
        node_y = circle_center[1] + radius * np.sin(angle)
        tangent_velocity = np.array([-np.sin(angle), np.cos(angle)])
        if tangent_velocity[0] < 0:
            tangent_velocity = -tangent_velocity
        boundary_nodes.append(Node('environmentBoundary', pos=(node_x, node_y), velocity=tangent_velocity))
    return boundary_nodes

def save_nodes_to_json(nodes, folder='NodeInitialLocationData', filename='nodes.json'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    all_nodes = [node.to_dict() for node in nodes]
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as json_file:
        json.dump(all_nodes, json_file, indent=4)
    print(f'Node data saved to {filepath}')

def reindex_nodes_inplace(nodes):
    for (new_id, node) in enumerate(nodes):
        node.id = int(new_id)
    Node._id_counter = len(nodes)

def generate_nodes_within_circle(circle_center, radius, node_counts, min_relay_dist=30, swarm_type=None, group_indices=None):
    generated_nodes = []
    node_types = ['relay', 'sensing', 'action', 'allround']
    group_indices = group_indices or {}
    for node_type in node_types:
        num_nodes = node_counts.get(node_type, 0)
        indices = list(group_indices.get(node_type, []))
        idx_pos = 0
        if swarm_type and node_type in swarm_type:
            ability = swarm_type[node_type][1]
        else:
            ability = 1
        for _ in range(num_nodes):
            placed = False
            while not placed:
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, radius * 3 / 10)
                x = circle_center[0] + r * np.cos(angle)
                y = circle_center[1] + r * np.sin(angle)
                if node_type == 'relay':
                    placed = all((np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2) > min_relay_dist for node in generated_nodes if node.type == 'relay')) and all((np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2) > 5 for node in generated_nodes))
                else:
                    placed = all((np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2) > 5 for node in generated_nodes))
            group_index = None
            if indices:
                group_index = indices[idx_pos % len(indices)]
                idx_pos += 1
            node = Node(type=node_type, pos=(x, y), group_index=group_index)
            node.ability = ability
            generated_nodes.append(node)
    return generated_nodes

def generate_and_save_nodes_crossing(base_type_name, swarm_count, node_folder='NodeInitialLocationData', taskinfo_folder='TaskInfo', num_samples=10, sensitivity_types=None, base_seed=None):
    """
    Generate nodes for the crossing scene and save both node states and task info.
    Args:
        base_type_name: swarm type name used for counts/abilities.
        swarm_count: scale multiplier passed to generate_large_swarm.
        node_folder: folder to save node json.
        taskinfo_folder: folder to save task info.
        num_samples: number of base samples to generate.
        sensitivity_types: list of dicts with keys name, relay_range, action_range, sensing_range.
        base_seed: optional seed to keep environments identical across combinations.
    """
    if sensitivity_types is None:
        sensitivity_types = sensitivity_types_get()
    edge_length = 5
    results = []
    for i in range(1, num_samples + 1):
        Node._id_counter = 0
        if base_seed is not None:
            reset_random_seed(base_seed + i)
        (x, y_upper, y_lower, circle_center, radius, channel_x, channel_y_upper, channel_y_lower) = generate_channel()
        obstacles = generate_obstacles(channel_x, channel_y_upper, channel_y_lower, num_obstacles=40)
        base_counts = dict(swarm_type_get()[base_type_name]) if hasattr(swarm_type_get(), '__getitem__') else {}
        node_dicts = generate_large_swarm(circle_center, base_type_name, swarm_count, edge_length)
        counts = {}
        group_indices = {}
        for d in node_dicts:
            node_type = d['type']
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'group_index' in d:
                group_indices.setdefault(node_type, []).append(d['group_index'])
        generated_nodes = generate_nodes_within_circle(circle_center, radius, counts, min_relay_dist=30, swarm_type=base_counts, group_indices=group_indices)
        obstacle_nodes = generate_obstacle_nodes(obstacles)
        boundary_nodes = generate_boundary_nodes(x, y_upper, y_lower)
        circular_boundary_nodes = generate_circular_boundary_nodes(circle_center, radius)
        circular_boundary_nodes = [n for n in circular_boundary_nodes if n.pos[0] < -float(L)]
        base_nodes = generated_nodes + boundary_nodes + circular_boundary_nodes + obstacle_nodes
        x_list = x.tolist()
        y_upper_list = y_upper.tolist()
        y_lower_list = y_lower.tolist()
        channel_data = {'x': x_list, 'y_upper': y_upper_list, 'y_lower': y_lower_list}
        for s_type in sensitivity_types:
            relay_range = s_type['relay_range']
            action_range = s_type['action_range']
            sensing_range = s_type.get('sensing_range')
            type_label = s_type['name']
            nodes_variant = clone_nodes_with_ranges(base_nodes, relay_range, action_range, sensing_range)
            reindex_nodes_inplace(nodes_variant)
            env_count = sum((1 for n in nodes_variant if 'environment' in n.type))
            non_env_count = len(nodes_variant) - env_count
            os.makedirs(node_folder, exist_ok=True)
            node_filename = f'crossing_{type_label}_{i}.json'
            save_nodes_to_json(nodes_variant, folder=node_folder, filename=node_filename)
            filepath = os.path.join(node_folder, node_filename)
            os.makedirs(taskinfo_folder, exist_ok=True)
            taskinfo_filename = os.path.join(taskinfo_folder, f'crossing_{type_label}_{i}.json')
            with open(taskinfo_filename, 'w') as json_file:
                json.dump(channel_data, json_file)
            print(f'[{type_label}_{i}] 数据已保存至:\n - {node_folder}/{node_filename}\n - {taskinfo_filename}\n节点统计：环境节点 {env_count} 个，非环境节点 {non_env_count} 个，总计 {len(nodes_variant)} 个')
            results.append({'filepath': filepath, 'filename': node_filename, 'relay_range': relay_range, 'action_range': action_range, 'sensing_range': sensing_range, 'type_label': type_label})
    return results
