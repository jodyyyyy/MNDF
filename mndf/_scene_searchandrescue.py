"""Numerical scene helpers extracted from SceneInitializationSearchAndRescue.py; plotting excluded."""

import copy
import json
import os
import random
import numpy as np
from ._scene_parameters import swarm_type_get, sensitivity_types_get
from ._scene_swarm import generate_large_swarm
from ._scene_parameters import generate_fixed_parameters_searchandresque


class Node:
    _id_counter = 0

    def __init__(self, type, pos=(0, 0), velocity=(0, 0), task=None, ability=1, group_index=None):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.type = type
        self.pos = np.array(pos)
        self.velocity = np.array(velocity)
        self.task = task
        self.parameters = generate_fixed_parameters_searchandresque()
        self.ability = ability
        self.group_index = group_index
        self.initialize()

    def initialize(self):
        if self.type not in ('relay', 'sensing', 'action', 'allround') and 'environment' not in self.type:
            raise ValueError(f"Invalid type: {self.type}. Type must be 'relay', 'sensing', 'action', or contain 'environment'.")
        initial_pos_range = self.parameters.get('initial_pos_range', {}).get(f'{self.type}', None)
        if np.all(self.pos == 0):
            pos = generate_random_position(initial_pos_range)
            self.pos = np.array(pos)
        if 'environment' in self.type:
            self.velocity = self.velocity * self.parameters.get('velocity_shill', 1)
        else:
            self.acceleration_max = self.parameters[f'initial_acceleration_max'][f'{self.type}']
            self.velocity_max = self.parameters[f'initial_velocity_max'][f'{self.type}']
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

def generate_nodes_on_circle(angle_interval=1):
    radius = generate_fixed_parameters_searchandresque()[f'boundary_range']
    nodes = []
    for angle in range(0, 360, angle_interval):
        theta = np.radians(angle)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        velocity_x = -x
        velocity_y = -y
        magnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        if magnitude != 0:
            velocity_x /= magnitude
            velocity_y /= magnitude
        node = Node(type='environmentBoundary', pos=(x, y), velocity=(velocity_x, velocity_y))
        nodes.append(node)
    return nodes

def generate_random_position(initial_pos_range):
    if isinstance(initial_pos_range, list) and len(initial_pos_range) == 2:
        (min_radius, max_radius) = initial_pos_range
    else:
        max_radius = initial_pos_range
        min_radius = 0
    radius = random.uniform(min_radius, max_radius)
    theta = random.uniform(0, 360)
    x = radius * np.cos(np.radians(theta))
    y = radius * np.sin(np.radians(theta))
    return (x, y)

def save_nodes_to_json(nodes, folder='NodeInitialLocationData', filename='nodes.json'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    all_nodes = [node.to_dict() for node in nodes]
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as json_file:
        json.dump(all_nodes, json_file, indent=4)
    print(f'Node data saved to {filepath}')

def _pick_relay_angle(relay_idx, sector_bounds):
    (start_deg, end_deg) = sector_bounds[relay_idx % len(sector_bounds)]
    angle_deg = np.random.uniform(start_deg, end_deg)
    return np.deg2rad(angle_deg)

def generate_nodes_within_circle(circle_center, radius, node_counts, min_relay_dist=50, swarm_type=None, group_indices=None):
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
        for relay_idx in range(num_nodes):
            placed = False
            while not placed:
                if node_type == 'relay':
                    angle = _pick_relay_angle(relay_idx, [(0.0, 120.0), (120.0, 240.0), (240.0, 360.0)])
                else:
                    angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, radius)
                x = circle_center[0] + r * np.cos(angle)
                y = circle_center[1] + r * np.sin(angle)
                if node_type == 'relay':
                    placed = all((np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2) > min_relay_dist for node in generated_nodes if node.type == 'relay')) and all((np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2) > 5 for node in generated_nodes))
                else:
                    placed = all((np.sqrt((x - node.pos[0]) ** 2 + (y - node.pos[1]) ** 2) > 5 for node in generated_nodes))
            vec_from_center = np.array([x - circle_center[0], y - circle_center[1]])
            direction = vec_from_center / (np.linalg.norm(vec_from_center) + 1e-08)
            velocity = tuple(direction * 5)
            group_index = None
            if indices:
                group_index = indices[idx_pos % len(indices)]
                idx_pos += 1
            new_node = Node(type=node_type, pos=(x, y), velocity=velocity, group_index=group_index)
            new_node.ability = ability
            generated_nodes.append(new_node)
    for (node_type, count) in node_counts.items():
        if node_type not in node_types:
            for _ in range(count):
                new_node = Node(node_type)
                generated_nodes.append(new_node)
    return generated_nodes

def generate_and_save_nodes_searchandrescue(circle_center, base_type_name, swarm_count, node_folder='NodeInitialLocationData', taskinfo_folder='TaskInfo', environment_task_count=60, num_samples=10, sensitivity_types=None, base_seed=None):
    """
    Generate nodes for search-and-rescue and save both node states and task info.
    Args:
        circle_center: center coordinate for swarm generation.
        base_type_name: swarm type name used for counts/abilities.
        swarm_count: scale multiplier passed to generate_large_swarm.
        node_folder: folder to save node json.
        taskinfo_folder: folder to save task info.
        environment_task_count: number of environmentTask nodes.
        num_samples: number of base samples to generate.
        sensitivity_types: list of dicts with keys name, relay_range, action_range, sensing_range.
        base_seed: optional seed to keep environments identical across combinations.
    """
    if sensitivity_types is None:
        sensitivity_types = sensitivity_types_get()
    range_area = generate_fixed_parameters_searchandresque()[f'boundary_range']
    base_counts = dict(swarm_type_get()[base_type_name])
    edge_length = 8
    results = []
    for i in range(1, num_samples + 1):
        Node._id_counter = 0
        if base_seed is not None:
            reset_random_seed(base_seed + i)
        node_dicts = generate_large_swarm(circle_center, base_type_name, swarm_count, edge_length)
        counts = {}
        group_indices = {}
        for d in node_dicts:
            node_type = d['type']
            counts[node_type] = counts.get(node_type, 0) + 1
            if 'group_index' in d:
                group_indices.setdefault(node_type, []).append(d['group_index'])
        nodes = generate_nodes_within_circle(circle_center, radius=50.0, node_counts=counts, min_relay_dist=50, swarm_type=base_counts, group_indices=group_indices)
        env_task_nodes = [Node(type='environmentTask', task=10) for _ in range(environment_task_count)]
        boundary_nodes = generate_nodes_on_circle()
        base_nodes = nodes + env_task_nodes + boundary_nodes
        task_info = {'range': range_area, 'node_count': len(nodes)}
        for s_type in sensitivity_types:
            relay_range = s_type['relay_range']
            action_range = s_type['action_range']
            sensing_range = s_type.get('sensing_range')
            type_label = s_type['name']
            nodes_variant = clone_nodes_with_ranges(base_nodes, relay_range, action_range, sensing_range)
            env_count = sum((1 for n in nodes_variant if 'environment' in n.type))
            non_env_count = len(nodes_variant) - env_count
            os.makedirs(node_folder, exist_ok=True)
            filename = f'searchandrescue_{type_label}_{i}.json'
            save_nodes_to_json(nodes_variant, folder=node_folder, filename=filename)
            filepath = os.path.join(node_folder, filename)
            os.makedirs(taskinfo_folder, exist_ok=True)
            taskinfo_filename = os.path.join(taskinfo_folder, filename)
            with open(taskinfo_filename, 'w') as json_file:
                json.dump(task_info, json_file)
            print(f'[{type_label}_{i}] 数据已保存至:\n - {node_folder}/{filename}\n - {taskinfo_filename}\n节点统计：环境节点 {env_count} 个，非环境节点 {non_env_count} 个，总计 {len(nodes_variant)} 个')
            results.append({'filepath': filepath, 'filename': filename, 'relay_range': relay_range, 'action_range': action_range, 'sensing_range': sensing_range, 'type_label': type_label})
    return results
