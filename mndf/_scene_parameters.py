"""Numerical scene helpers extracted from ParametersGetFix.py; plotting excluded."""



def generate_fixed_parameters_cross():
    fixed_parameters = {'control_points_upper': [[0, 160], [200, 240], [400, 160], [600, 240], [800, 160]], 'control_points_lower': [[0, 0], [200, 80], [400, 0], [600, 80], [800, 0]], 'velocity_shill': 10, 'initial_velocity_max': {'relay': 10, 'sensing': 10, 'action': 10, 'environment': 10, 'allround': 10}, 'initial_acceleration_max': {'relay': 5, 'sensing': 5, 'action': 5, 'environment': 0, 'allround': 5}, 'initial_range_communication': {'sensing': 30, 'relay': 30, 'action': 30, 'allround': 30}, 'initial_range_sensing': {'sensing': 30, 'allround': 30}, 'initial_range_action': {'action': 10, 'allround': 10}}
    return fixed_parameters

def generate_fixed_parameters_searchandresque():
    fixed_parameters = {'boundary_range': 300, 'velocity_shill': 10, 'initial_pos_range': {'relay': 50, 'sensing': 50, 'action': 50, 'allround': 50, 'environmentTask': [150, 250]}, 'initial_velocity_max': {'relay': 10, 'sensing': 10, 'action': 10, 'environment': 10, 'allround': 10}, 'initial_acceleration_max': {'relay': 5, 'sensing': 5, 'action': 5, 'environment': 0, 'allround': 5}, 'initial_range_communication': {'sensing': 30, 'relay': 30, 'action': 30, 'allround': 30}, 'initial_range_sensing': {'sensing': 30, 'allround': 30}, 'initial_range_action': {'action': 10, 'allround': 10}}
    return fixed_parameters

def swarm_type_get():
    """基础集群类型（仅节点数量与能力），敏感性组合单独在 sensitivity_types_get 配置。"""
    return {'heter_1': {'action': [10, 1], 'sensing': [10, 1], 'relay': [1, 1], 'allround': [0, 1]}}

def sensitivity_types_get():
    """敏感性分析的6种集群类型（仅覆盖通信/行动范围）。"""
    combos = []
    for rc in (30, 60, 90):
        for ra in (5, 10, 15):
            for rs in (15, 30, 45):
                combos.append({'name': f'heter_1_rc{rc}_ra{ra}_rs{rs}', 'relay_range': rc, 'action_range': ra, 'sensing_range': rs})
    return combos
