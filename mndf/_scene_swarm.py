"""Numerical scene helpers extracted from ModularNodeGeneration.py; plotting excluded."""

import math
from ._scene_parameters import swarm_type_get

def regular_polygon_points(center, radius, num_points, start_angle_rad=0):
    """返回正多边形上均匀布置的顶点坐标（顺时针）"""
    (cx, cy) = center
    points = []
    for i in range(num_points):
        angle = start_angle_rad + 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points

def generate_nodes_in_hexagon(center, swarm_type_name, edge_length, group_index=None):
    """
    若节点总数为10/11或20/21，则使用结构化正多边形均匀布点，否则保持原始随机采样逻辑。
    """
    (x_center, y_center) = center
    swarm_type = swarm_type_get()
    node_distribution = swarm_type.get(swarm_type_name)
    if node_distribution is None:
        raise ValueError(f"Swarm type '{swarm_type_name}' not found.")
    total_count = sum([v[0] for v in node_distribution.values()])
    relay_count = node_distribution.get('relay', [0])[0]
    nodes = []
    if total_count in [10, 11, 20, 21]:
        remaining = total_count
        if total_count % 2 == 1:
            nodes.append({'type': 'relay', 'pos': (x_center, y_center), 'group_index': group_index})
            remaining -= 1
        alpha = 1.618
        k1 = round(remaining / (1.0 + alpha))
        k2 = remaining - k1
        R1_intra = k1 * edge_length / (2 * math.pi)
        R2_intra = k2 * edge_length / (2 * math.pi)
        R1 = max(R1_intra, R2_intra / alpha)
        R2 = alpha * R1
        delta = math.pi / max(k1, k2)

        def d_cross(R1, R2):
            return math.sqrt(R1 ** 2 + R2 ** 2 - 2 * R1 * R2 * math.cos(delta))
        dc = d_cross(R1, R2)
        if dc < edge_length:
            s = edge_length / dc
            R1 *= s
            R2 *= s
        pts1 = regular_polygon_points(center, R1, k1)
        pts2 = regular_polygon_points(center, R2, k2)
        pts = pts1 + pts2
        type_counts = [(k, v[0]) for (k, v) in node_distribution.items() if k != 'relay']
        type_queue = {k: v for (k, v) in type_counts}
        type_order = [k for (k, _) in type_counts]
        type_cycle = []
        i = 0
        while sum(type_queue.values()) > 0:
            k = type_order[i % len(type_order)]
            if type_queue[k] > 0:
                type_cycle.append(k)
                type_queue[k] -= 1
            i += 1
        for (i, (x, y)) in enumerate(pts):
            nodes.append({'type': type_cycle[i], 'pos': (x, y), 'group_index': group_index})
        return nodes

def swarm_range_get(edge_length):
    kapa1 = 1.8
    kapa2 = 1.8
    swarm_range = {'heter_1': edge_length * math.sqrt(kapa1 * 20 / 3), 'heter_1_2_rc': edge_length * math.sqrt(kapa1 * 20 / 3), 'heter_1_3_rc': edge_length * math.sqrt(kapa1 * 20 / 3), 'homo_1': edge_length * math.sqrt(kapa2 * 20 / 3)}
    return swarm_range

def hex_grid_centers(center, radius, count):
    """
    以 30° 为起点的六边形蜂窝排布，生成 count 个中心坐标。
    - count==6：只生成一圈(6个点)，不含中心
    - count==7：先生成一圈(6个点)，最后追加中心
    生成完成后，将几何中心平移到给定的 center。
    适用于 flat-topped 六边形（边水平对齐）。
    """
    (cx, cy) = center
    if count <= 0:
        return []
    spacing = math.sqrt(3) * radius
    angles_deg = [30, 90, 150, 210, 270, 330]
    directions = [(spacing * math.cos(math.radians(a)), spacing * math.sin(math.radians(a))) for a in angles_deg]
    if count == 6:
        pts = directions[:]
    elif count == 7:
        pts = directions[:]
        pts.append((0.0, 0.0))
    else:
        pts = [(0.0, 0.0)]
        if count > 1:
            layer = 1
            while len(pts) < count:
                (dx0, dy0) = directions[4]
                x = layer * dx0
                y = layer * dy0
                for d in range(6):
                    (dx, dy) = directions[d]
                    for _ in range(layer):
                        if len(pts) >= count:
                            break
                        pts.append((x, y))
                        x += dx
                        y += dy
                    if len(pts) >= count:
                        break
                layer += 1
        pts = pts[:count]
    mx = sum((p[0] for p in pts)) / len(pts)
    my = sum((p[1] for p in pts)) / len(pts)
    (dx, dy) = (cx - mx, cy - my)
    pts = [(x + dx, y + dy) for (x, y) in pts]
    return pts

def generate_large_swarm(center, swarm_type_name, swarm_count, edge_length):
    """
    生成一个大 swarm，由多个小 swarm 组成
    - 每个 swarm 的六边形中心通过蜂窝排布得到
    - 每个 swarm 内部结构由 swarm_type_name 决定
    """
    range_dict = swarm_range_get(edge_length)
    radius = range_dict.get(swarm_type_name)
    if radius is None:
        raise ValueError(f"Swarm type '{swarm_type_name}' not defined in swarm_range.")
    centers = hex_grid_centers(center, radius, swarm_count)
    all_nodes = []
    for (group_index, swarm_center) in enumerate(centers):
        nodes = generate_nodes_in_hexagon(swarm_center, swarm_type_name, edge_length, group_index=group_index)
        all_nodes.extend(nodes)
    return all_nodes
