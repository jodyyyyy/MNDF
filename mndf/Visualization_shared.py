import json
import matplotlib.pyplot as plt
from ._visualization_colors import color_matching_get
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Polygon
import os
import numpy as np
import torch
import networkx as nx


def _save_figure_with_svg(fig, save_path, dpi=300, save_svg=False, save_pdf=False):
    save_path = str(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    root, ext = os.path.splitext(save_path)
    ext = ext.lower()
    if ext == ".svg":
        png_path = f"{root}.png"
        svg_path = save_path
        pdf_path = f"{root}.pdf"
    elif ext == ".pdf":
        png_path = f"{root}.png"
        svg_path = f"{root}.svg"
        pdf_path = save_path
    else:
        png_path = save_path
        svg_path = f"{root}.svg"
        pdf_path = f"{root}.pdf"
    fig.savefig(png_path, dpi=dpi)
    if save_svg or ext == ".svg":
        fig.savefig(svg_path, format="svg")
    if save_pdf or ext == ".pdf":
        fig.savefig(pdf_path, format="pdf")


def _build_edge_lookup(net):
    """
    从 net.edge_index_by_net 中构建查找表：
    {net_name: [(src_type_name, dst_type_name, src_global_ids_cpu, dst_global_ids_cpu), ...]}
    """
    type_id = net.type_id_tensor
    edge_lookup = {}
    for net_name, edge_index in net.edge_index_by_net.items():
        if edge_index is None or edge_index.numel() == 0:
            continue
        src, dst = edge_index
        src_types = type_id[src]
        dst_types = type_id[dst]
        pairs = torch.stack([src_types, dst_types], dim=0)
        unique_pairs = torch.unique(pairs, dim=1)
        for st, dt in unique_pairs.T:
            mask = (src_types == st) & (dst_types == dt)
            src_gid = src[mask].detach().cpu()
            dst_gid = dst[mask].detach().cpu()
            if src_gid.numel() == 0:
                continue
            src_name = net.id_to_type_name[int(st)]
            dst_name = net.id_to_type_name[int(dt)]
            edge_lookup.setdefault(net_name, []).append((src_name, dst_name, src_gid, dst_gid))
    return edge_lookup


def _iter_edge_pairs(edge_lookup, net_name):
    for src_type, dst_type, src_ids, dst_ids in edge_lookup.get(net_name, []):
        for s, d in zip(src_ids.tolist(), dst_ids.tolist()):
            yield src_type, dst_type, s, d


def _tensor_to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _extract_type_names_from_state_data(state_data):
    if isinstance(state_data, dict):
        if "type_names" in state_data:
            return list(state_data["type_names"])
        if "id_to_type_name" in state_data:
            id_to_name = state_data["id_to_type_name"]
            max_id = max(int(k) for k in id_to_name.keys())
            names = [None] * (max_id + 1)
            for k, v in id_to_name.items():
                names[int(k)] = v
            return names
        if "type_name_to_id" in state_data:
            name_to_id = state_data["type_name_to_id"]
            max_id = max(int(v) for v in name_to_id.values())
            names = [None] * (max_id + 1)
            for name, tid in name_to_id.items():
                names[int(tid)] = name
            return names
    return None


def visualize_initial_distribution(nodes_data, scene_type_id=1, figsize=(12, 9), save_path=None, dpi=300, save_svg=False, save_pdf=False):
    node_colors = color_matching_get()
    if isinstance(nodes_data, (str, os.PathLike)):
        with open(nodes_data, "r", encoding="utf-8") as f:
            nodes = json.load(f)
    else:
        nodes = nodes_data

    node_positions_all = {}
    finished_nodes_all = []
    for node in nodes:
        node_type = node.get("type", "unknown")
        pos = node.get("pos", [0.0, 0.0])
        task_val = node.get("task", -1)
        if task_val is None:
            task_val = -1
        if task_val == 0:
            finished_nodes_all.append((f"{node_type}Finished", pos))
        else:
            node_positions_all.setdefault(node_type, []).append(pos)
    for ntype, pos in finished_nodes_all:
        node_positions_all.setdefault(ntype, []).append(pos)
    for ntype in node_positions_all.keys():
        if "Finished" in ntype and ntype not in node_colors:
            node_colors[ntype] = "#888888"

    if scene_type_id == 1:
        alpha = 0
        beta = -0.3
        size = 10
    elif scene_type_id == 2:
        alpha = 0.05
        beta = -0.05
        size = 10
    else:
        alpha = 0
        beta = 0
        size = 10

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title("All Node Positions")
    ax.set_aspect("equal")
    for node_type, pos_list in node_positions_all.items():
        pos_arr = np.array(pos_list, dtype=float)
        if pos_arr.size > 0:
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors.get(node_type, "#888888"), s=size)
    ax.legend(loc="upper left", bbox_to_anchor=(alpha, beta), ncol=2, markerscale=2.5, fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _save_figure_with_svg(plt.gcf(), save_path, dpi=dpi, save_svg=save_svg, save_pdf=save_pdf)
        plt.close()
        print(f"[save] initial distribution saved to: {save_path}")
    else:
        plt.show()


def visualize_initial_two_panels(nodes_data, scene_type_id=1, figsize=(14, 6), save_path=None, dpi=300, save_svg=False, save_pdf=False):
    node_colors = color_matching_get()
    if isinstance(nodes_data, (str, os.PathLike)):
        with open(nodes_data, "r", encoding="utf-8") as f:
            nodes = json.load(f)
    else:
        nodes = nodes_data

    node_positions_all = {}
    finished_nodes_all = []
    for node in nodes:
        node_type = node.get("type", "unknown")
        pos = node.get("pos", [0.0, 0.0])
        task_val = node.get("task", -1)
        if task_val is None:
            task_val = -1
        if task_val == 0:
            finished_nodes_all.append((f"{node_type}Finished", pos))
        else:
            node_positions_all.setdefault(node_type, []).append(pos)
    for ntype, pos in finished_nodes_all:
        node_positions_all.setdefault(ntype, []).append(pos)
    for ntype in node_positions_all.keys():
        if "Finished" in ntype and ntype not in node_colors:
            node_colors[ntype] = "#888888"

    if scene_type_id == 1:
        alpha = 0
        beta = -0.3
        size = 10
    elif scene_type_id == 2:
        alpha = 0.05
        beta = -0.05
        size = 10
    else:
        alpha = 0
        beta = 0
        size = 10

    node_positions_non_env = {
        k: v for k, v in node_positions_all.items() if "environment" not in k
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].set_title("All Nodes")
    axes[1].set_title("Non-Environment Nodes")
    for ax in axes:
        ax.set_aspect("equal")

    for node_type, pos_list in node_positions_all.items():
        pos_arr = np.array(pos_list, dtype=float)
        if pos_arr.size > 0:
            axes[0].scatter(
                pos_arr[:, 0],
                pos_arr[:, 1],
                label=node_type,
                color=node_colors.get(node_type, "#888888"),
                s=size,
            )

    for node_type, pos_list in node_positions_non_env.items():
        pos_arr = np.array(pos_list, dtype=float)
        if pos_arr.size > 0:
            axes[1].scatter(
                pos_arr[:, 0],
                pos_arr[:, 1],
                label=node_type,
                color=node_colors.get(node_type, "#888888"),
                s=size,
            )

    axes[0].legend(loc="upper left", bbox_to_anchor=(alpha, beta), ncol=2, markerscale=2.5, fontsize=12)
    axes[1].legend(loc="upper left", bbox_to_anchor=(alpha, beta), ncol=2, markerscale=2.5, fontsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _save_figure_with_svg(plt.gcf(), save_path, dpi=dpi, save_svg=save_svg, save_pdf=save_pdf)
        plt.close()
        print(f"[save] two-panel initial distribution saved to: {save_path}")
    else:
        plt.show()


def visualize_net(net, scene_type_id, color_scheme_id=6, figsize=(16, 12), save_path=None, dpi=300, save_svg=False, save_pdf=False):
    node_colors = color_matching_get()
    node_all_types = net.node_type_names
    node_types = [ntype for ntype in node_all_types if "environment" not in ntype]
    buffer = 30
    state_np = net.state.detach().cpu().numpy()
    task_np = net.task_tensor.detach().cpu().numpy()
    type_np = net.type_id_tensor.detach().cpu().numpy()

    edge_lookup = _build_edge_lookup(net)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i in range(5):
        axes[i].set_box_aspect(1)

    # 全部节点 + Finished
    node_positions_all = {}
    finished_nodes_all = []
    for node_type in node_all_types:
        tid = net.type_name_to_id[node_type]
        mask = type_np == tid
        node_ids = np.where(mask)[0]
        node_pos = state_np[node_ids, :2]
        task_values = task_np[node_ids]
        for pos, task_val, gid in zip(node_pos, task_values, node_ids):
            if task_val == 0:
                finished_nodes_all.append((f"{node_type}Finished", pos))
            else:
                node_positions_all.setdefault(node_type, []).append(pos)
    for ntype, pos in finished_nodes_all:
        node_positions_all.setdefault(ntype, []).append(pos)
    for ntype in node_positions_all.keys():
        if "Finished" in ntype:
            node_colors[ntype] = "#888888"
    # 非环境节点（全部保留，不看连通性）
    node_positions = {k: v for k, v in node_positions_all.items() if "environment" not in k}

    # 1) 仅节点
    ax = axes[0]
    ax.set_title("Node Positions (No Edges)")
    for node_type, pos_list in node_positions.items():
        pos_arr = np.array(pos_list)
        if pos_arr.size > 0:
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type])
    ax.set_aspect("equal")

    # 2) communication_net
    ax = axes[1]
    ax.set_title("communication_net")
    ax.set_aspect("equal")
    for node_type, pos_list in node_positions.items():
        pos_arr = np.array(pos_list)
        if pos_arr.size > 0:
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type])
    for _, _, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, "communication_net"):
        p1 = state_np[src_gid, :2]
        p2 = state_np[dst_gid, :2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=0.5, alpha=0.5)

    # 3) logic_net
    ax = axes[2]
    ax.set_title("logic_net")
    ax.set_aspect("equal")
    min_x, max_x, min_y, max_y = float("inf"), float("-inf"), float("inf"), float("-inf")
    connected_environment_nodes = set()
    for src_type, dst_type, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, "logic_net"):
        if "environment" in src_type:
            connected_environment_nodes.add(src_gid)
        if "environment" in dst_type:
            connected_environment_nodes.add(dst_gid)
    labeled_node_types = set()
    for node_type in node_all_types:
        tid = net.type_name_to_id[node_type]
        mask = type_np == tid
        node_ids = np.where(mask)[0]
        pos = state_np[node_ids, :2]
        if "environment" in node_type:
            for gid, p in zip(node_ids, pos):
                if gid in connected_environment_nodes:
                    if node_type not in labeled_node_types:
                        ax.scatter(p[0], p[1], label=node_type, color=node_colors[node_type])
                        labeled_node_types.add(node_type)
                    else:
                        ax.scatter(p[0], p[1], color=node_colors[node_type])
        else:
            if node_type in node_positions_all:
                pos_arr = np.array(node_positions_all[node_type])
                if pos_arr.size > 0:
                    ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type])
                    min_x = min(min_x, pos_arr[:, 0].min())
                    max_x = max(max_x, pos_arr[:, 0].max())
                    min_y = min(min_y, pos_arr[:, 1].min())
                    max_y = max(max_y, pos_arr[:, 1].max())

    ax.set_xlim(min_x - buffer, max_x + buffer)
    ax.set_ylim(min_y - buffer, max_y + buffer)
    for _, _, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, "logic_net"):
        p1 = state_np[src_gid, :2]
        p2 = state_np[dst_gid, :2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=0.5, alpha=0.5)

    # 4-5) sensing/action
    edge_types = ["sensing_net", "action_net"]
    for idx, net_type in enumerate(edge_types):
        ax = axes[idx + 3]
        ax.set_title(f"{net_type}")
        ax.set_aspect("equal")
        min_x, max_x, min_y, max_y = float("inf"), float("-inf"), float("inf"), float("-inf")
        connected_environment_nodes = set()
        for src_type, dst_type, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, net_type):
            if "environment" in src_type:
                connected_environment_nodes.add(src_gid)
            if "environment" in dst_type:
                connected_environment_nodes.add(dst_gid)
        labeled_node_types = set()
        for node_type in node_all_types:
            tid = net.type_name_to_id[node_type]
            mask = type_np == tid
            node_ids = np.where(mask)[0]
            pos = state_np[node_ids, :2]
            if "environment" in node_type:
                for gid, p in zip(node_ids, pos):
                    if gid in connected_environment_nodes:
                        if node_type not in labeled_node_types:
                            ax.scatter(p[0], p[1], label=node_type, color=node_colors[node_type])
                            labeled_node_types.add(node_type)
                        else:
                            ax.scatter(p[0], p[1], color=node_colors[node_type])
            else:
                if node_type in node_positions_all:
                    pos_arr = np.array(node_positions_all[node_type])
                    if pos_arr.size > 0:
                        ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type])
                        min_x = min(min_x, pos_arr[:, 0].min())
                        max_x = max(max_x, pos_arr[:, 0].max())
                        min_y = min(min_y, pos_arr[:, 1].min())
                        max_y = max(max_y, pos_arr[:, 1].max())
        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)
        for _, _, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, net_type):
            p1 = state_np[src_gid, :2]
            p2 = state_np[dst_gid, :2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=0.5, alpha=0.5)

    # 6) 全部
    ax = axes[5]
    ax.set_title("All Node Positions")
    ax.set_aspect("equal")
    if scene_type_id == 1:
        alpha = 0
        beta = -0.3
        size = 10
    elif scene_type_id == 2:
        alpha = 0.05
        beta = -0.05
        size = 10
    else:
        alpha = 0
        beta = 0
        size = 10
    for node_type, pos_list in node_positions_all.items():
        pos_arr = np.array(pos_list)
        if pos_arr.size > 0:
            ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type], s=size)
    ax.legend(loc="upper left", bbox_to_anchor=(alpha, beta), ncol=2, markerscale=2.5, fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _save_figure_with_svg(plt.gcf(), save_path, dpi=dpi, save_svg=save_svg, save_pdf=save_pdf)
        plt.close()
        print(f"[保存成功] 图像已保存至：{save_path}")
    else:
        plt.show()


def visualize_state_trajectories(
    state_data,
    color_scheme_id=6,
    max_speed=10.0,
    linewidth=1.0,
    marker_size=60,
    trace_linewidth=None,
    figsize=(14, 10),
    stage_count=4,
    dt=None,
    config_stage_indices=None,
    save_path=None,
    dpi=300,
    save_svg=False,
    save_pdf=False,
):
    node_colors = color_matching_get()
    base_colors = [
        "#0ebeff",
        "#29b0f7",
        "#44a2ee",
        "#5e95e6",
        "#7987dd",
        "#9479d5",
        "#af6bcc",
        "#c95ec4",
        "#e450bb",
        "#ff42b3",
    ]
    if isinstance(state_data, (str, os.PathLike)):
        data = torch.load(state_data, map_location="cpu")
    else:
        data = state_data

    if not isinstance(data, dict):
        raise ValueError("state_data must be a dict or a path to a saved dict.")
    scene_type_id = int(data.get("scene_type_id", 1))
    metric_series = data.get("metric_series")
    metric_dt = data.get("metric_dt", dt)

    for key in ("pos", "vel", "task", "type_id"):
        if key not in data:
            raise ValueError("state_data must contain pos, vel, task, and type_id.")
    pos = _tensor_to_numpy(data["pos"])
    vel = _tensor_to_numpy(data["vel"])
    task = _tensor_to_numpy(data["task"])
    type_id = _tensor_to_numpy(data["type_id"]).astype(int)
    group_index = None
    if "group_index" in data:
        group_index = _tensor_to_numpy(data["group_index"]).astype(int)
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("pos must have shape [T, N, 2].")
    if vel.ndim != 3 or vel.shape[2] != 2:
        raise ValueError("vel must have shape [T, N, 2].")
    if task.ndim != 2:
        raise ValueError("task must have shape [T, N].")

    type_names = _extract_type_names_from_state_data(data)
    if type_names is None:
        max_tid = int(type_id.max()) if type_id.size > 0 else -1
        type_names = [f"type_{i}" for i in range(max_tid + 1)]

    steps, node_count = pos.shape[0], pos.shape[1]
    if vel.shape[0] != steps or vel.shape[1] != node_count:
        raise ValueError("vel shape does not match pos.")
    if task.shape[0] != steps or task.shape[1] != node_count:
        raise ValueError("task shape does not match pos.")
    if type_id.shape[0] != node_count:
        raise ValueError("type_id length does not match pos/vel node count.")
    if group_index is not None and group_index.shape[0] != node_count:
        raise ValueError("group_index length does not match pos/vel node count.")

    speed = np.linalg.norm(vel, axis=2)
    max_speed = float(max_speed)
    if not np.isfinite(max_speed) or max_speed <= 0:
        max_speed = float(np.nanmax(speed)) if np.isfinite(speed).any() else 1.0
    max_speed = max(max_speed, 1e-6)
    speed_norm = plt.Normalize(vmin=0.0, vmax=max_speed)
    speed_cmap = LinearSegmentedColormap.from_list("trajectory_speed_gradient", base_colors)

    def _normalize_rgba(color_value):
        rgba = np.array(to_rgba(color_value), dtype=float)
        if np.nanmax(rgba[:3]) > 1.0:
            rgba[:3] = rgba[:3] / 255.0
        rgba[:3] = np.clip(rgba[:3], 0.0, 1.0)
        rgba[3] = np.clip(rgba[3], 0.0, 1.0)
        return rgba

    def _stage_title(end_idx):
        if dt is None:
            return f"stage @ {end_idx}"
        return f"t = {float(end_idx) * float(dt):.1f} s"

    def _metric_symbol(metric_name):
        scene_suffix = "t" if scene_type_id == 1 else "s"
        metric_map = {
            "phi_collision": r"$\phi^{\mathrm{coll}}$",
            "phi_correlation": r"$\phi^{\mathrm{corr}}$",
            "phi_task": rf"$\phi^{{\mathrm{{task}}}}_{{\mathrm{{{scene_suffix}}}}}$",
            "phi_gather": rf"$\phi^{{\mathrm{{gather}}}}_{{\mathrm{{{scene_suffix}}}}}$",
        }
        return metric_map.get(metric_name, metric_name)

    def _legend_label(raw_label):
        label = str(raw_label)
        if label.startswith("$"):
            return label
        if label == "Task Finished":
            return label
        label = label.replace("_", " ").replace("-", " ")
        words = []
        for part in label.split():
            current = []
            for char in part:
                if current and char.isupper() and (not current[-1].isupper()):
                    words.append("".join(current))
                    current = [char]
                else:
                    current.append(char)
            if current:
                words.append("".join(current))
        return " ".join(word[:1].upper() + word[1:] for word in words if word)

    def _apply_adaptive_legend(ax, fontsize=8, handles=None, labels=None):
        legend_kwargs = {
            "loc": "best",
            "prop": {"family": "Times New Roman", "size": fontsize},
            "frameon": True,
            "fancybox": True,
            "framealpha": 0.82,
            "borderaxespad": 0.3,
            "handlelength": 1.4,
        }
        if handles is None and labels is None:
            legend = ax.legend(**legend_kwargs)
        else:
            legend = ax.legend(handles, [_legend_label(label) for label in labels], **legend_kwargs)
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor((1.0, 1.0, 1.0, 0.82))
            frame.set_edgecolor("#bbbbbb")
            frame.set_linewidth(0.6)
        return legend

    non_env_mask = np.array(
        [
            False
            if tid < 0 or tid >= len(type_names) or type_names[tid] is None
            else ("environment" not in str(type_names[tid]))
            for tid in type_id
        ],
        dtype=bool,
    )
    if not np.any(non_env_mask):
        non_env_mask = np.ones((node_count,), dtype=bool)

    if scene_type_id == 2:
        task_node_mask = np.array(
            [
                False
                if tid < 0 or tid >= len(type_names) or type_names[tid] is None
                else (str(type_names[tid]) == "environmentTask")
                for tid in type_id
            ],
            dtype=bool,
        )
        if not np.any(task_node_mask):
            task_node_mask = np.any(task > 0.0, axis=0)
    else:
        task_node_mask = non_env_mask & np.any(~np.isclose(task, 0.0), axis=0)
        if not np.any(task_node_mask):
            task_node_mask = non_env_mask.copy()

    effective_last_idx = steps - 1
    for ti in range(steps):
        if scene_type_id == 1:
            stage_mask = non_env_mask & (~np.isclose(task[ti], 0.0))
            if not np.any(stage_mask):
                stage_mask = non_env_mask
            x_vals = pos[ti, stage_mask, 0]
            x_vals = x_vals[np.isfinite(x_vals)]
            if x_vals.size > 0 and float(np.min(x_vals)) > 800.0:
                effective_last_idx = ti
                break
        elif scene_type_id == 2:
            if np.any(task_node_mask) and np.all(np.isclose(task[ti, task_node_mask], 0.0)):
                effective_last_idx = ti
                break
    effective_end_time = float(effective_last_idx) * float(dt) if dt is not None else float(effective_last_idx)

    stage_count = int(stage_count)
    if stage_count <= 0:
        raise ValueError("stage_count must be a positive integer.")
    boundary_values = np.linspace(0, effective_last_idx, num=stage_count + 1)
    stage_end_indices = []
    for idx in boundary_values:
        idx_int = min(max(int(round(idx)), 0), effective_last_idx)
        if not stage_end_indices or idx_int != stage_end_indices[-1]:
            stage_end_indices.append(idx_int)
    if not stage_end_indices:
        stage_end_indices = [effective_last_idx]
    if stage_end_indices[-1] != effective_last_idx:
        stage_end_indices.append(effective_last_idx)

    if config_stage_indices is None:
        selected_stage_indices = list(range(1, max(len(stage_end_indices) - 1, 1)))
    else:
        selected_stage_indices = []
        for item in config_stage_indices:
            idx = int(item)
            if idx < 0:
                continue
            if idx > len(stage_end_indices) - 1:
                continue
            selected_stage_indices.append(idx)
    selected_stage_indices = sorted(set(selected_stage_indices))

    if scene_type_id == 2:
        task_alpha_values = np.abs(task[task > 0.0])
        task_alpha_scale = float(np.nanmax(task_alpha_values)) if task_alpha_values.size > 0 else 1.0
    else:
        task_alpha_scale = float(np.nanmax(np.abs(task[:, non_env_mask]))) if np.any(non_env_mask) else 1.0
    if not np.isfinite(task_alpha_scale) or task_alpha_scale <= 1e-9:
        task_alpha_scale = 1.0

    def _task_alpha_at(end_idx, node_id):
        if scene_type_id != 2:
            return 1.0
        raw_task_value = float(task[end_idx, node_id])
        if raw_task_value < 0.0:
            return 1.0
        if np.isclose(raw_task_value, 0.0):
            return 0.0
        task_value = abs(raw_task_value)
        return float(np.clip(0.2 + 0.8 * (task_value / task_alpha_scale), 0.2, 1.0))

    def _completed_task_in_window(start_idx, end_idx, node_id):
        if scene_type_id != 2:
            return False
        start_task_value = float(task[start_idx, node_id])
        end_task_value = float(task[end_idx, node_id])
        return start_task_value > 0.0 and np.isclose(end_task_value, 0.0)

    def _draw_trajectory(
        ax,
        end_idx,
        start_idx=0,
        show_colorbar=True,
        show_legend=True,
        fixed_limits=None,
    ):
        ax.set_aspect("equal")
        start_idx = min(max(int(start_idx), 0), end_idx)

        all_segments = []
        all_colors = []
        env_points_by_type = {}

        active_mask = non_env_mask & (~np.isclose(task[end_idx], 0.0))
        if not np.any(active_mask):
            active_mask = non_env_mask.copy()

        unique_types = np.unique(type_id)
        for tid in unique_types:
            tid_int = int(tid)
            type_name = (
                type_names[tid_int] if (tid_int < len(type_names) and type_names[tid_int]) else f"type_{tid_int}"
            )
            node_idx_all = np.where(type_id == tid_int)[0]
            if node_idx_all.size == 0:
                continue
            base_color = node_colors.get(type_name, "#888888")

            if "environment" in type_name:
                pts = pos[end_idx, node_idx_all, :]
                finite_mask = np.isfinite(pts).all(axis=1)
                if finite_mask.any():
                    visible_pts = []
                    visible_colors = []
                    visible_sizes = []
                    for point, node_id in zip(pts[finite_mask], node_idx_all[finite_mask]):
                        node_id = int(node_id)
                        completed_in_window = _completed_task_in_window(start_idx, end_idx, node_id)
                        rgba = _normalize_rgba("#888888" if completed_in_window else base_color)
                        alpha = 1.0 if completed_in_window else _task_alpha_at(end_idx, node_id)
                        if alpha <= 0.0:
                            continue
                        rgba[3] = alpha
                        visible_pts.append(point)
                        visible_colors.append(rgba)
                        visible_sizes.append(marker_size * 0.5 if completed_in_window else marker_size)
                    if visible_pts:
                        env_points_by_type[type_name] = (np.asarray(visible_pts), visible_colors, visible_sizes)
                continue

            node_idx = node_idx_all[active_mask[node_idx_all]]
            if node_idx.size == 0:
                continue

            base_rgba = _normalize_rgba(base_color)
            min_color_mix = 0.6
            light_rgb = np.clip(base_rgba[:3] * min_color_mix + (1.0 - min_color_mix), 0.0, 1.0)
            for nid in node_idx:
                p = pos[start_idx : end_idx + 1, nid, :]
                if p.shape[0] < 2:
                    continue
                segs = np.stack([p[:-1], p[1:]], axis=1)
                seg_speed = 0.5 * (speed[start_idx:end_idx, nid] + speed[start_idx + 1 : end_idx + 1, nid])
                finite_mask = np.isfinite(segs).all(axis=(1, 2)) & np.isfinite(seg_speed)
                if not finite_mask.any():
                    continue
                segs = segs[finite_mask]
                seg_speed = seg_speed[finite_mask]
                if scene_type_id == 2:
                    segment_colors = speed_cmap(speed_norm(seg_speed))
                    segment_colors[:, 3] *= _task_alpha_at(end_idx, nid)
                    all_colors.append(segment_colors)
                else:
                    all_colors.append(speed_cmap(speed_norm(seg_speed)))
                all_segments.append(segs)

        if all_segments:
            segs = np.concatenate(all_segments, axis=0)
            colors = np.concatenate(all_colors, axis=0)
            lw = float(trace_linewidth) if trace_linewidth is not None else float(linewidth)
            if scene_type_id == 2:
                lw *= 0.6
            lc = LineCollection(segs, colors=colors, linewidths=lw)
            ax.add_collection(lc)

        for type_name, (pts, colors, sizes) in env_points_by_type.items():
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, c=colors, label=type_name, marker="o", zorder=5)

        plot_type_ids = list(np.unique(type_id))
        if scene_type_id == 2:
            plot_type_ids.sort(
                key=lambda tid: 1
                if (
                    int(tid) < len(type_names)
                    and type_names[int(tid)]
                    and type_names[int(tid)] == "relay"
                )
                else 0
            )

        for tid in plot_type_ids:
            tid_int = int(tid)
            type_name = (
                type_names[tid_int] if (tid_int < len(type_names) and type_names[tid_int]) else f"type_{tid_int}"
            )
            node_idx = np.where(type_id == tid_int)[0]
            if node_idx.size > 0:
                if scene_type_id == 2:
                    node_idx = np.asarray(
                        [
                            int(nid)
                            for nid in node_idx
                            if active_mask[int(nid)] or _completed_task_in_window(start_idx, end_idx, int(nid))
                        ],
                        dtype=int,
                    )
                else:
                    node_idx = node_idx[active_mask[node_idx]]
            if node_idx.size == 0:
                continue
            pts = pos[end_idx, node_idx, :]
            finite_mask = np.isfinite(pts).all(axis=1)
            if not finite_mask.any():
                continue
            if "environment" in type_name:
                continue
            base_color = node_colors.get(type_name, "#888888")
            point_colors = []
            visible_pts = []
            visible_sizes = []
            for point, node_id in zip(pts[finite_mask], node_idx[finite_mask]):
                node_id = int(node_id)
                completed_in_window = _completed_task_in_window(start_idx, end_idx, node_id)
                rgba = _normalize_rgba("#888888" if completed_in_window else base_color)
                alpha = 1.0 if completed_in_window else _task_alpha_at(end_idx, node_id)
                if alpha <= 0.0:
                    continue
                rgba[3] = alpha
                visible_pts.append(point)
                point_colors.append(rgba)
                visible_sizes.append(0.5 if completed_in_window else 1.0)
            if not visible_pts:
                continue
            visible_pts = np.asarray(visible_pts)
            node_marker_size = marker_size
            if scene_type_id == 2:
                node_marker_size = max(float(marker_size) * 0.2, 8.0)
            point_sizes = [node_marker_size * scale for scale in visible_sizes]
            ax.scatter(
                visible_pts[:, 0],
                visible_pts[:, 1],
                s=point_sizes,
                c=point_colors,
                edgecolors="none",
                linewidths=0.0,
                label=type_name,
                zorder=6,
            )

        if fixed_limits is not None:
            min_x, max_x, min_y, max_y = [float(value) for value in fixed_limits]
        else:
            pts = pos[start_idx : end_idx + 1, :, :].reshape(-1, 2)
            finite = np.isfinite(pts).all(axis=1)
            if finite.any():
                min_x = float(np.min(pts[finite, 0]))
                max_x = float(np.max(pts[finite, 0]))
                min_y = float(np.min(pts[finite, 1]))
                max_y = float(np.max(pts[finite, 1]))
            else:
                min_x = min_y = -1.0
                max_x = max_y = 1.0
            buffer = 30.0
            min_x -= buffer
            max_x += buffer
            min_y -= buffer
            max_y += buffer
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                dedup = dict(zip(labels, handles))
                _apply_adaptive_legend(ax, fontsize=9, handles=list(dedup.values()), labels=list(dedup.keys()))

        if show_colorbar and all_segments:
            sm = plt.cm.ScalarMappable(cmap=speed_cmap, norm=speed_norm)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
            cbar.set_label("Speed (m/s)", fontsize=12)
            cbar.ax.tick_params(labelsize=12)

    def _draw_configuration(ax, end_idx):
        ax.set_aspect("equal")
        active_mask = non_env_mask & (~np.isclose(task[end_idx], 0.0))
        if not np.any(active_mask):
            active_mask = non_env_mask

        finite_mask = active_mask & np.isfinite(pos[end_idx]).all(axis=1)
        if not np.any(finite_mask):
            ax.axis("off")
            return None

        pts = pos[end_idx, finite_mask, :]
        tids = type_id[finite_mask]
        centroid = np.mean(pts, axis=0)
        rel_pts = pts - centroid

        for tid in np.unique(tids):
            tid_int = int(tid)
            type_name = (
                type_names[tid_int] if (tid_int < len(type_names) and type_names[tid_int]) else f"type_{tid_int}"
            )
            mask = tids == tid_int
            ax.scatter(
                rel_pts[mask, 0],
                rel_pts[mask, 1],
                s=max(marker_size * 0.6, 20),
                c=[_normalize_rgba(node_colors.get(type_name, "#888888"))],
                edgecolors="none",
                linewidths=0.0,
            )

        span = np.max(np.abs(rel_pts), axis=0) if rel_pts.size > 0 else np.array([1.0, 1.0], dtype=float)
        radius = max(float(np.max(span)), 1.0) * 1.25
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#666666")
        return centroid

    def _stage_anchor(end_idx):
        active_mask = non_env_mask & (~np.isclose(task[end_idx], 0.0))
        if not np.any(active_mask):
            active_mask = non_env_mask
        finite_mask = active_mask & np.isfinite(pos[end_idx]).all(axis=1)
        if not np.any(finite_mask):
            return None
        pts = pos[end_idx, finite_mask, :]
        return np.mean(pts, axis=0)

    if save_path:
        base_dir = os.path.dirname(save_path)
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    has_metrics = isinstance(metric_series, dict) and len(metric_series) > 0
    metric_items = [(name, list(series)) for name, series in metric_series.items()] if has_metrics else []

    def _metric_slice(series, start_idx, end_idx, dt_for_metric):
        if not series:
            return [], []
        if dt_for_metric > 0 and dt is not None:
            start_time = float(start_idx) * float(dt)
            end_time = float(end_idx) * float(dt)
            start_len = int(np.floor(start_time / dt_for_metric + 1e-9))
            end_len = int(np.floor(end_time / dt_for_metric + 1e-9)) + 1
        else:
            start_len = int(start_idx)
            end_len = int(end_idx) + 1
        start_len = max(0, min(len(series), start_len))
        end_len = max(start_len + 1, min(len(series), end_len))
        series_cut = series[start_len:end_len]
        time_axis = [i * dt_for_metric for i in range(start_len, end_len)]
        return time_axis, series_cut

    def _draw_metric_row(fig, metric_grid, start_idx, end_idx):
        dt_for_metric = float(metric_dt) if metric_dt is not None else 1.0
        for metric_idx, (name, series) in enumerate(metric_items):
            time_axis, series_cut = _metric_slice(series, start_idx, end_idx, dt_for_metric)
            ax = fig.add_subplot(metric_grid[0, metric_idx])
            if time_axis and series_cut:
                ax.plot(time_axis, series_cut, linewidth=1.2, label=_metric_symbol(name))
            ax.set_xlabel("time", fontsize=12)
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=8)
            if time_axis:
                ax.set_xlim(time_axis[0], time_axis[-1])
            if ax.get_legend_handles_labels()[0]:
                _apply_adaptive_legend(ax, fontsize=12)

    def _draw_output_figure(start_idx, end_idx, inset_stage_indices, output_path, title=None):
        inset_count = len(inset_stage_indices)
        fig = plt.figure(figsize=figsize)
        outer_heights = []
        if inset_count > 0:
            outer_heights.append(1.1)
        outer_heights.append(3.0)
        colorbar_row = len(outer_heights)
        outer_heights.append(0.18)
        if has_metrics:
            outer_heights.append(0.9)
        outer = fig.add_gridspec(len(outer_heights), 1, height_ratios=outer_heights)
        outer_row = 0

        if inset_count > 0:
            inset_grid = outer[outer_row, 0].subgridspec(1, inset_count)
            outer_row += 1
        else:
            inset_grid = None

        main_ax = fig.add_subplot(outer[outer_row, 0])
        outer_row += 1

        _draw_trajectory(main_ax, end_idx, start_idx=start_idx, show_colorbar=False, show_legend=True)
        main_ax.set_title(title or _stage_title(end_idx))

        if inset_count > 0:
            for inset_col, stage_idx in enumerate(inset_stage_indices):
                inset_end_idx = stage_end_indices[stage_idx]
                inset_ax = fig.add_subplot(inset_grid[0, inset_col])
                _draw_configuration(inset_ax, inset_end_idx)
                inset_ax.set_title(_stage_title(inset_end_idx), fontsize=10, pad=4)
                anchor = _stage_anchor(inset_end_idx)
                if anchor is not None:
                    main_ax.axvline(anchor[0], color="#888888", linestyle="--", linewidth=0.8, alpha=0.8)
                    connector = ConnectionPatch(
                        xyA=(0.5, 0.0),
                        coordsA=inset_ax.transAxes,
                        xyB=(anchor[0], anchor[1]),
                        coordsB=main_ax.transData,
                        color="#666666",
                        linewidth=0.8,
                        alpha=0.9,
                    )
                    fig.add_artist(connector)

        sm = plt.cm.ScalarMappable(cmap=speed_cmap, norm=speed_norm)
        sm.set_array([])
        cax = fig.add_subplot(outer[colorbar_row, 0])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Speed (m/s)", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        if has_metrics:
            metric_grid = outer[colorbar_row + 1, 0].subgridspec(1, len(metric_items), wspace=0.22)
            _draw_metric_row(fig, metric_grid, start_idx, end_idx)

        fig.tight_layout()
        if output_path:
            _save_figure_with_svg(fig, output_path, dpi=dpi, save_svg=save_svg, save_pdf=save_pdf)
            plt.close(fig)
            print(f"[save] trajectory image saved to: {output_path}")
        else:
            plt.show()

    def _draw_sa_segments_figure(stage_indices, output_path):
        segment_bounds = []
        previous_stage_idx = 0
        for stage_idx in stage_indices:
            start_idx = stage_end_indices[previous_stage_idx]
            end_idx = stage_end_indices[stage_idx]
            if end_idx >= start_idx:
                segment_bounds.append((stage_idx, start_idx, end_idx))
            previous_stage_idx = stage_idx
        if not segment_bounds:
            segment_bounds = [(len(stage_end_indices) - 1, 0, effective_last_idx)]
        has_task_finished_in_segments = any(
            _completed_task_in_window(start_idx, end_idx, node_id)
            for _, start_idx, end_idx in segment_bounds
            for node_id in range(node_count)
        )

        segment_count = len(segment_bounds)
        if segment_count <= 1:
            row_counts = [segment_count]
        else:
            first_row_count = int(np.floor(segment_count / 2))
            second_row_count = segment_count - first_row_count
            row_counts = [first_row_count, second_row_count]
        rows = len(row_counts)
        grid_cols = max(row_counts)
        sa_figsize = (figsize[0], figsize[1] + 2) if has_metrics else figsize
        fig = plt.figure(figsize=sa_figsize)
        outer_heights = [3.0] * rows
        colorbar_row = rows
        outer_heights.append(0.18)
        if has_metrics:
            outer_heights.append(2.0)
        outer = fig.add_gridspec(len(outer_heights), 1, height_ratios=outer_heights)

        trajectory_axes = []
        trajectory_axis_meta = []
        panel_index = 0
        for row_idx, cols in enumerate(row_counts):
            slot_cols = grid_cols * 2
            panel_grid = outer[row_idx, 0].subgridspec(1, slot_cols, wspace=0.18)
            row_start_slot = int((slot_cols - cols * 2) // 2)
            for col_idx in range(cols):
                slot = row_start_slot + col_idx * 2
                ax = fig.add_subplot(panel_grid[0, slot : slot + 2])
                stage_idx, start_idx, end_idx = segment_bounds[panel_index]
                _draw_trajectory(
                    ax,
                    end_idx,
                    start_idx=start_idx,
                    show_colorbar=False,
                    show_legend=False,
                    fixed_limits=(-320.0, 320.0, -320.0, 320.0),
                )
                ax.set_title(f"{_stage_title(start_idx)} -> {_stage_title(end_idx)}", fontsize=10)
                trajectory_axes.append(ax)
                trajectory_axis_meta.append((ax, row_idx, col_idx))
                panel_index += 1

        for ax, row_idx, col_idx in trajectory_axis_meta:
            show_y_axis = col_idx == 0
            show_x_axis = True
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(show_y_axis)
            ax.spines["bottom"].set_visible(show_x_axis)
            ax.tick_params(
                axis="y",
                left=show_y_axis,
                labelleft=show_y_axis,
                right=False,
                labelright=False,
            )
            ax.tick_params(
                axis="x",
                bottom=show_x_axis,
                labelbottom=show_x_axis,
                top=False,
                labeltop=False,
            )

        legend_handles = []
        legend_labels = []
        for ax in trajectory_axes:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels)
        if legend_handles or has_task_finished_in_segments:
            dedup = dict(zip(legend_labels, legend_handles))
            legend_labels_order = list(dedup.keys())
            if has_task_finished_in_segments:
                legend_labels_order.append("Task Finished")
            opaque_handles = []
            for label in legend_labels_order:
                rgba = _normalize_rgba("#888888" if label == "Task Finished" else node_colors.get(label, "#888888"))
                rgba[3] = 1.0
                opaque_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="None",
                        marker="o",
                        markersize=8,
                        markerfacecolor=rgba,
                        markeredgecolor="none",
                        markeredgewidth=0.0,
                    )
                )
            legend = fig.legend(
                opaque_handles,
                [_legend_label(label) for label in legend_labels_order],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
                ncol=min(len(legend_labels_order), 6),
                prop={"family": "Times New Roman", "size": 12},
                frameon=True,
                fancybox=True,
                framealpha=0.82,
                markerscale=1.4,
                handlelength=2.0,
            )
            frame = legend.get_frame()
            frame.set_facecolor((1.0, 1.0, 1.0, 0.82))
            frame.set_edgecolor("#bbbbbb")
            frame.set_linewidth(0.6)

        if trajectory_axes:
            sm = plt.cm.ScalarMappable(cmap=speed_cmap, norm=speed_norm)
            sm.set_array([])
            cax = fig.add_subplot(outer[colorbar_row, 0])
            cbar = fig.colorbar(
                sm,
                cax=cax,
                orientation="horizontal",
            )
            cbar.set_label("Speed (m/s)", fontsize=12)
            cbar.ax.tick_params(labelsize=12)

        if has_metrics:
            metric_grid = outer[colorbar_row + 1, 0].subgridspec(1, len(metric_items), wspace=0.3)
            _draw_metric_row(fig, metric_grid, 0, effective_last_idx)

        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        if output_path:
            _save_figure_with_svg(fig, output_path, dpi=dpi, save_svg=save_svg, save_pdf=save_pdf)
            plt.close(fig)
            print(f"[save] trajectory image saved to: {output_path}")
        else:
            plt.show()

    if scene_type_id == 2:
        sa_stage_indices = selected_stage_indices or [len(stage_end_indices) - 1]
        _draw_sa_segments_figure(sa_stage_indices, save_path)
        return

    _draw_output_figure(
        0,
        effective_last_idx,
        selected_stage_indices,
        save_path,
        title=_stage_title(effective_last_idx),
    )


def visualize_net_confront(net, color_scheme_id=6, figsize=(28, 12), save_path=None, dpi=300, save_svg=False, save_pdf=False):
    """
    对抗场景（scene_type_id=3），2x7 子图，保留原绘制逻辑但使用全局张量。
    """
    node_colors = color_matching_get()
    node_all_types = net.node_type_names

    state_np = net.state.detach().cpu().numpy()
    task_np = net.task_tensor.detach().cpu().numpy()
    type_np = net.type_id_tensor.detach().cpu().numpy()
    edge_lookup = _build_edge_lookup(net)

    original_group = [ntype for ntype in node_all_types if "environment" not in ntype and "enemy" not in ntype]
    enemy_group = [ntype for ntype in node_all_types if "enemy" in ntype]
    environment_types = [ntype for ntype in node_all_types if "environment" in ntype]

    fig, axes = plt.subplots(2, 7, figsize=figsize)
    for i in range(2):
        for j in range(7):
            if j != 6:
                axes[i, j].set_box_aspect(1)

    groups = [
        {"name": "Original Group", "types": original_group, "row": 0},
        {"name": "Enemy Group", "types": enemy_group, "row": 1},
    ]

    # 预处理每组节点/finished
    all_groups_node_positions = {}
    all_groups_finished_nodes = {}
    for group in groups:
        group_key = group["name"]
        group_types = group["types"]

        connected_ids_group = set()
        net_types_to_check = ["enemy_communication_net", "enemy_logic_net"] if group_types == enemy_group else [
            "communication_net",
            "logic_net",
        ]
        for net_type in net_types_to_check:
            for src_type_edge, dst_type_edge, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, net_type):
                if src_type_edge in group_types or dst_type_edge in group_types:
                    connected_ids_group.add(src_gid)
                    connected_ids_group.add(dst_gid)

        node_positions_group = {}
        finished_nodes_group = []
        for node_type in group_types:
            tid = net.type_name_to_id[node_type]
            mask = type_np == tid
            node_ids = np.where(mask)[0]
            node_pos = state_np[node_ids, :2]
            task_values = task_np[node_ids]
            for pos, task_val, gid in zip(node_pos, task_values, node_ids):
                if gid in connected_ids_group:
                    if task_val == 0:
                        finished_nodes_group.append((f"{node_type}Finished", pos))
                    else:
                        node_positions_group.setdefault(node_type, []).append(pos)
        for ntype, pos in finished_nodes_group:
            node_positions_group.setdefault(ntype, []).append(pos)
        all_groups_node_positions[group_key] = node_positions_group
        all_groups_finished_nodes[group_key] = finished_nodes_group
        for ntype in node_positions_group:
            if "Finished" in ntype:
                node_colors[ntype] = "#555555" if group_types == original_group else "#555555"

    # 绘制
    for group in groups:
        group_key = group["name"]
        group_types = group["types"]
        row_idx = group["row"]

        # 7 个网络：communication/logic/sensing/action + 3 占位/对比
        net_names = [
            "communication_net" if group_types == original_group else "enemy_communication_net",
            "logic_net" if group_types == original_group else "enemy_logic_net",
            "sensing_net" if group_types == original_group else "enemy_sensing_net",
            "action_net" if group_types == original_group else "enemy_action_net",
            None,
            None,
            None,
        ]

        for col_idx, net_name in enumerate(net_names):
            ax = axes[row_idx, col_idx]
            if net_name is None:
                ax.axis("off")
                continue
            ax.set_box_aspect(1)
            ax.set_title(f"{group_key} - {net_name}")

            node_pos_map = all_groups_node_positions[group_key]
            for node_type, pos_list in node_pos_map.items():
                pos_arr = np.array(pos_list)
                if pos_arr.size > 0:
                    ax.scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type])

            for src_type, dst_type, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, net_name):
                p1 = state_np[src_gid, :2]
                p2 = state_np[dst_gid, :2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=0.5, alpha=0.5)

    # 第 7 列：两组对比
    ax_comp = axes[:, 6]
    for i, group in enumerate(groups):
        node_pos_map = all_groups_node_positions[group["name"]]
        for node_type, pos_list in node_pos_map.items():
            pos_arr = np.array(pos_list)
            if pos_arr.size > 0:
                ax_comp[i].scatter(pos_arr[:, 0], pos_arr[:, 1], label=node_type, color=node_colors[node_type])
        ax_comp[i].set_title(f"{group['name']} Nodes")
        ax_comp[i].set_aspect("equal")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _save_figure_with_svg(plt.gcf(), save_path, dpi=dpi, save_svg=save_svg, save_pdf=save_pdf)
        plt.close()
        print(f"[保存成功] 图像已保存至：{save_path}")
    else:
        plt.show()


def visualize_net_all_with_tadpole_tails(
    net,
    scene_type_id,
    tail_scale=2.0,
    tail_base_width=2.0,
    alpha_tail=0.35,
    marker_size=20,
    figsize=(12, 10),
    save_path=None,
):
    """
    单图总览 + 蝌蚪尾巴，改为使用全局张量。
    """
    node_colors = color_matching_get()
    node_all_types = net.node_type_names

    state_np = net.state.detach().cpu().numpy()
    task_np = net.task_tensor.detach().cpu().numpy()
    type_np = net.type_id_tensor.detach().cpu().numpy()

    node_positions_all = {}
    finished_nodes_all = []
    for node_type in node_all_types:
        tid = net.type_name_to_id[node_type]
        mask = type_np == tid
        node_ids = np.where(mask)[0]
        pos = state_np[node_ids, :2]
        vel = state_np[node_ids, 2:4]
        task_val = task_np[node_ids]
        for p, v, t, gid in zip(pos, vel, task_val, node_ids):
            if t == 0:
                finished_nodes_all.append((f"{node_type}Finished", gid, p, v))
            else:
                node_positions_all.setdefault(node_type, []).append((gid, p, v))
    for ntype, gid, p, v in finished_nodes_all:
        node_positions_all.setdefault(ntype, []).append((gid, p, v))
    for ntype in list(node_positions_all.keys()):
        if "Finished" in ntype and ntype not in node_colors:
            node_colors[ntype] = "#888888"

    all_pts = []
    for items in node_positions_all.values():
        for _, p, _ in items:
            all_pts.append(p)
    if all_pts:
        pts = np.asarray(all_pts, dtype=float)
        min_x, max_x = float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0]))
        min_y, max_y = float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1]))
    else:
        min_x = min_y = -1.0
        max_x = max_y = 1.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title("All Node Positions (Tadpole Tails = Speed)")
    ax.set_aspect("equal")

    for node_type, items in node_positions_all.items():
        if "environment" in node_type:
            continue
        color = node_colors.get(node_type, "#888888")
        for _, p, v in items:
            p = np.asarray(p, dtype=float)
            v = np.asarray(v, dtype=float)
            if p.shape != (2,) or v.shape != (2,):
                continue
            if not (np.isfinite(p).all() and np.isfinite(v).all()):
                continue
            speed = float(np.hypot(v[0], v[1]))
            if speed <= 1e-8:
                continue
            L = tail_scale * speed
            v_hat = v / speed
            n_hat = np.array([-v_hat[1], v_hat[0]])
            base_left = p + tail_base_width * n_hat
            base_right = p - tail_base_width * n_hat
            tip = p - L * v_hat
            tri = np.vstack([base_left, base_right, tip])
            tail_poly = Polygon(tri, closed=True, facecolor=color, edgecolor="none", alpha=alpha_tail, antialiased=True)
            ax.add_patch(tail_poly)
    for node_type, items in node_positions_all.items():
        color = node_colors.get(node_type, "#888888")
        if not items:
            continue
        pts = np.array([p for _, p, _ in items], dtype=float)
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, label=node_type, color=color, zorder=3)

    buffer = 30.0
    ax.set_xlim(min_x - buffer, max_x + buffer)
    ax.set_ylim(min_y - buffer, max_y + buffer)
    if scene_type_id == 1:
        alpha_legend, beta_legend = 0, -0.3
    elif scene_type_id == 2:
        alpha_legend, beta_legend = 0.05, -0.05
    else:
        alpha_legend, beta_legend = 0, 0
    ax.legend(loc="upper left", bbox_to_anchor=(alpha_legend, beta_legend), ncol=2, markerscale=2.5, fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _save_figure_with_svg(plt.gcf(), save_path, dpi=300)
        plt.close()
        print(f"[保存成功] 图像已保存至：{save_path}")
    else:
        plt.show()


def visualize_subgroup_networks(net, figsize=(20, 15), save_path=None):
    node_colors = color_matching_get()
    node_all_types = net.node_type_names
    type_np = net.type_id_tensor.detach().cpu().numpy()
    task_np = net.task_tensor.detach().cpu().numpy()
    state_np = net.state.detach().cpu().numpy()
    edge_lookup = _build_edge_lookup(net)

    node_types = [ntype for ntype in node_all_types if "environment" not in ntype]
    all_nodes = []
    for ntype in node_types:
        tid = net.type_name_to_id[ntype]
        mask = type_np == tid
        all_nodes.extend(np.where(mask)[0].tolist())

    G = nx.Graph()
    for net_type in ["communication_net", "logic_net"]:
        for _, _, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, net_type):
            G.add_edge(src_gid, dst_gid)

    connected_components = list(nx.connected_components(G))
    if not connected_components:
        print("Warning: No connected components found!")
        return

    connected_components.sort(key=len, reverse=True)
    top_components = connected_components[:3]
    if len(top_components) == 1:
        top_components = [top_components[0]] * 3
    elif len(top_components) == 2:
        top_components = [top_components[0], top_components[1], top_components[1]]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    for i in range(4):
        axes[i].set_box_aspect(1)

    network_types = ["communication_net", "logic_net", "sensing_net", "action_net"]

    node_positions_all = {}
    finished_nodes = []
    for node_type in node_all_types:
        tid = net.type_name_to_id[node_type]
        mask = type_np == tid
        node_ids = np.where(mask)[0]
        node_pos = state_np[node_ids, :2]
        task_values = task_np[node_ids]
        for pos, task_val, gid in zip(node_pos, task_values, node_ids):
            if task_val == 0:
                finished_nodes.append((f"{node_type}Finished", pos))
            else:
                node_positions_all.setdefault(node_type, []).append(pos)
    for ntype, pos in finished_nodes:
        node_positions_all.setdefault(ntype, []).append(pos)
    for ntype in node_positions_all.keys():
        if "Finished" in ntype:
            node_colors[ntype] = "#888888"

    component_identifiers = []
    for component in top_components:
        if component:
            component_identifiers.append((min(component), component))
        else:
            component_identifiers.append((float("inf"), component))
    component_identifiers.sort(key=lambda x: x[0])
    sorted_components = [comp for _, comp in component_identifiers]

    for i, component in enumerate(sorted_components):
        ax = axes[i]
        component_id = min(component) if component else "Empty"
        ax.set_title(f"Subgroup {i+1} (ID: {component_id}, Size: {len(component)})")
        ax.set_aspect("equal", adjustable="box")

        component_nodes = set(component)
        related_environment_nodes = set()
        for net_type in network_types:
            for src_type_edge, dst_type_edge, src_gid, dst_gid in _iter_edge_pairs(edge_lookup, net_type):
                if src_gid in component_nodes and "environment" in dst_type_edge:
                    related_environment_nodes.add(dst_gid)
                if dst_gid in component_nodes and "environment" in src_type_edge:
                    related_environment_nodes.add(src_gid)

        labeled_node_types = set()
        for node_type in node_types:
            tid = net.type_name_to_id[node_type]
            mask = type_np == tid
            node_ids = np.where(mask)[0]
            node_pos = state_np[node_ids, :2]
            task_values = task_np[node_ids]
            for pos, gid, task_val in zip(node_pos, node_ids, task_values):
                if gid in component_nodes:
                    display_type = f"{node_type}Finished" if task_val == 0 else node_type
                    color = "#555555" if task_val == 0 else node_colors[node_type]
                    if display_type not in labeled_node_types:
                        ax.scatter(pos[0], pos[1], color=color, label=display_type, s=100, alpha=0.8)
                        labeled_node_types.add(display_type)
                    else:
                        ax.scatter(pos[0], pos[1], color=color, s=100, alpha=0.8)

        for node_type in node_all_types:
            if "environment" in node_type:
                tid = net.type_name_to_id[node_type]
                mask = type_np == tid
                node_ids = np.where(mask)[0]
                node_pos = state_np[node_ids, :2]
                task_values = task_np[node_ids]
                for pos, gid, task_val in zip(node_pos, node_ids, task_values):
                    if gid in related_environment_nodes:
                        display_type = f"{node_type}Finished" if task_val == 0 else node_type
                        color = "#555555" if task_val == 0 else node_colors[node_type]
                        if display_type not in labeled_node_types:
                            ax.scatter(pos[0], pos[1], color=color, label=display_type, s=80, alpha=0.6)
                            labeled_node_types.add(display_type)
                        else:
                            ax.scatter(pos[0], pos[1], color=color, s=80, alpha=0.6)

        all_x, all_y = [], []
        for node_type in node_all_types:
            if "environment" not in node_type:
                tid = net.type_name_to_id[node_type]
                mask = type_np == tid
                node_ids = np.where(mask)[0]
                node_pos = state_np[node_ids, :2]
                for pos, gid in zip(node_pos, node_ids):
                    if gid in component_nodes:
                        all_x.append(pos[0])
                        all_y.append(pos[1])
        if all_x and all_y:
            buffer = 30
            x_span = max(all_x) - min(all_x)
            y_span = max(all_y) - min(all_y)
            max_span = max(x_span, y_span) / 2 + buffer
            x_center = (max(all_x) + min(all_x)) / 2
            y_center = (max(all_y) + min(all_y)) / 2
            ax.set_xlim(x_center - max_span, x_center + max_span)
            ax.set_ylim(y_center - max_span, y_center + max_span)

    # 第4图：全网节点（无边）
    ax = axes[3]
    ax.set_title("Full Network Nodes (No Edges)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    for node_type in node_all_types:
        tid = net.type_name_to_id[node_type]
        mask = type_np == tid
        node_pos = state_np[mask, :2]
        task_values = task_np[mask]
        active_positions = []
        finished_positions = []
        for pos, task_val in zip(node_pos, task_values):
            if task_val == 0:
                finished_positions.append(pos)
            else:
                active_positions.append(pos)
        if active_positions:
            pts = np.array(active_positions)
            ax.scatter(pts[:, 0], pts[:, 1], label=node_type, color=node_colors.get(node_type, "#888888"), s=60)
        if finished_positions:
            pts = np.array(finished_positions)
            finished_type = f"{node_type}Finished"
            if finished_type not in node_colors:
                node_colors[finished_type] = "#555555"
            ax.scatter(pts[:, 0], pts[:, 1], label=finished_type, color=node_colors[finished_type], s=60, alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _save_figure_with_svg(plt.gcf(), save_path, dpi=300)
        plt.close()
        print(f"[保存成功] 图像已保存至：{save_path}")
    else:
        plt.show()
