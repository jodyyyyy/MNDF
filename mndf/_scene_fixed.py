"""Numerical scene helpers extracted from ScaleGeneralizationSearchAndRescue.py; plotting excluded."""



def format_float(value):
    return round(float(value), 6) if value is not None else None

def build_agent_node(node_type, pos, velocity, parameters, range_override=None, group_index=None):
    data = {'type': node_type, 'pos': [format_float(v) for v in pos], 'velocity': [format_float(v) for v in velocity], 'velocity_max': format_float(parameters['initial_velocity_max'][node_type]), 'acceleration_max': format_float(parameters['initial_acceleration_max'][node_type]), 'r_s': None, 'r_c': None, 'r_a': None, 'task': None}
    if range_override:
        rc = range_override.get('rc')
        ra = range_override.get('ra')
        rs = range_override.get('rs')
    else:
        rc = ra = rs = None
    if node_type == 'relay':
        data['r_c'] = format_float(rc if rc is not None else parameters['initial_range_communication'][node_type])
    elif node_type == 'sensing':
        data['r_s'] = format_float(rs if rs is not None else parameters['initial_range_sensing'][node_type])
        data['r_c'] = format_float(parameters['initial_range_communication'][node_type])
    elif node_type == 'action':
        data['r_a'] = format_float(ra if ra is not None else parameters['initial_range_action'][node_type])
        data['r_c'] = format_float(parameters['initial_range_communication'][node_type])
    elif node_type == 'allround':
        data['r_a'] = format_float(ra if ra is not None else parameters['initial_range_action'][node_type])
        data['r_s'] = format_float(rs if rs is not None else parameters['initial_range_sensing'][node_type])
        data['r_c'] = format_float(parameters['initial_range_communication'][node_type])
    if group_index is not None:
        data['group_index'] = int(group_index)
    return data
