"""Unified MNDF dynamics and fitness for Crossing and Search-and-Rescue."""

import json
import os
import math
from typing import Dict, List, Tuple

import torch



# Shared communication cap; task-specific drive is selected by scene_type_id.
RELAY_TOPK = 30
SELF_DRIVE_DIRECTION = (1.0, 0.0)


def get_relay_topk() -> int:
    """Return the fixed relay communication out-degree cap."""
    return RELAY_TOPK


def get_self_drive_direction() -> tuple[float, float]:
    """Return the fixed left-to-right Crossing self-drive direction."""
    return SELF_DRIVE_DIRECTION

IDX_POS = slice(0, 2)
IDX_VEL = slice(2, 4)
IDX_VMAX = 4
IDX_AMAX = 5
IDX_R_S = 6
IDX_R_C = 7
IDX_R_A = 8


def net_constraints_get(filepath):
    """
    Resolve constraint file based on the node config path (consistent with top-k batch logic).
    """
    base_dir = os.path.dirname(os.path.dirname(filepath))
    constraint_dir = os.path.join(base_dir, "Constraint")
    filename_core = os.path.basename(filepath)[:-5]
    parts = filename_core.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    constraint_filename = "_".join(parts) + ".json"
    constraint_path = os.path.join(constraint_dir, constraint_filename)
    if not os.path.exists(constraint_path):
        constraint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Constraint", constraint_filename)
    if not os.path.exists(constraint_path):
        raise FileNotFoundError(f"Constraint file not found: {constraint_path}")
    with open(constraint_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def parameter_range_get(filepath):
    """
    Resolve parameter file based on the node config path (same pattern as constraint resolution).
    """
    base_dir = os.path.dirname(os.path.dirname(filepath))
    param_dir = os.path.join(base_dir, "Parameter")
    filename_core = os.path.basename(filepath)[:-5]
    parts = filename_core.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    param_filename = "_".join(parts) + ".json"
    param_path = os.path.join(param_dir, param_filename)
    if not os.path.exists(param_path):
        param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Parameter", param_filename)
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Parameter file not found: {param_path}")
    with open(param_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


class Net:
    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.state = None
        self.task_tensor = None
        self.type_id_tensor = None
        self.valid_node_mask = None
        self.node_type_names = []
        self.type_name_to_id = {}
        self.id_to_type_name = {}
        self.type_to_ids = {}
        self.node_id_tensor = None
        self.features_tensor = None
        self.net_constraint = None
        self.feature_dim = 0
        self.state_b = None
        self.task_b = None
        self.group_index_tensor = None

    def load_nodes_from_json(self, filepath, scene_type_id):
        if isinstance(filepath, (list, tuple)):
            filepaths = list(filepath)
            if not filepaths:
                raise ValueError("filepath list is empty.")
            self._load_nodes_from_json_single(filepaths[0], scene_type_id)
            self._load_nodes_from_json_batch(filepaths)
            return
        self._load_nodes_from_json_single(filepath, scene_type_id)

    def _load_nodes_from_json_single(self, filepath, scene_type_id):
        with open(filepath, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        self.net_constraint = net_constraints_get(filepath)

        type_to_feats, type_to_ids, type_to_task = {}, {}, {}
        group_index_by_id = {}
        max_gid = -1
        for node in nodes_data:
            node_type, node_id = node["type"], node["id"]
            velocity_max = node.get("velocity_max", 0.0) or 0.0
            acceleration_max = node.get("acceleration_max", 0.0) or 0.0
            r_s = node.get("r_s", 0.0) or 0.0
            r_c = node.get("r_c", 0.0) or 0.0
            r_a = node.get("r_a", 0.0) or 0.0
            task = node.get("task", -1)
            if task is None:
                task = -1
            group_index = node.get("group_index", -1)
            if group_index is None:
                group_index = -1

            feat = torch.tensor(node["pos"], dtype=torch.float32)
            vel = torch.tensor(node["velocity"], dtype=torch.float32)
            extra = torch.tensor([velocity_max, acceleration_max, r_s, r_c, r_a], dtype=torch.float32)
            features = torch.cat([feat, vel, extra], dim=0)
            max_gid = max(max_gid, node_id)

            type_to_feats.setdefault(node_type, []).append(features)
            type_to_ids.setdefault(node_type, []).append(node_id)
            type_to_task.setdefault(node_type, []).append(task)
            group_index_by_id[node_id] = int(group_index)

        self.node_type_names = list(type_to_feats.keys())
        self.type_name_to_id = {name: i for i, name in enumerate(self.node_type_names)}
        self.id_to_type_name = {i: name for name, i in self.type_name_to_id.items()}

        feat_dim = 9
        state = torch.full((max_gid + 1, feat_dim), float("nan"), device=self.device, dtype=torch.float32)
        task_tensor = torch.full((max_gid + 1,), -1.0, device=self.device, dtype=torch.float32)
        type_id_tensor = torch.full((max_gid + 1,), -1, device=self.device, dtype=torch.long)
        valid_mask = torch.zeros((max_gid + 1,), device=self.device, dtype=torch.bool)
        group_index_tensor = torch.full((max_gid + 1,), -1, device=self.device, dtype=torch.long)

        self.type_to_ids = {}
        for node_type, features in type_to_feats.items():
            ids_tensor = torch.tensor(type_to_ids[node_type], dtype=torch.long, device=self.device)
            feats_tensor = torch.stack([f.to(self.device) for f in features])
            tasks_tensor = torch.tensor(type_to_task[node_type], dtype=torch.float32, device=self.device)
            state[ids_tensor] = feats_tensor
            task_tensor[ids_tensor] = tasks_tensor
            type_id = self.type_name_to_id[node_type]
            type_id_tensor[ids_tensor] = type_id
            valid_mask[ids_tensor] = True
            self.type_to_ids[node_type] = ids_tensor
        for gid, gval in group_index_by_id.items():
            group_index_tensor[gid] = int(gval)

        self.state = state
        self.task_tensor = task_tensor
        self.type_id_tensor = type_id_tensor
        self.valid_node_mask = valid_mask
        self.group_index_tensor = group_index_tensor
        self.node_id_tensor = torch.arange(max_gid + 1, device=self.device, dtype=torch.long)
        self.features_tensor = self.state
        self.feature_dim = feat_dim
        self.state_b = None
        self.task_b = None

    def _build_state_from_nodes_data(self, nodes_data):
        max_gid = 0
        feat_dim = self.feature_dim
        for node in nodes_data:
            node_id = node["id"]
            max_gid = max(max_gid, node_id)

        state = torch.full((max_gid + 1, feat_dim), float("nan"), device=self.device, dtype=torch.float32)
        task_tensor = torch.full((max_gid + 1,), -1.0, device=self.device, dtype=torch.float32)
        type_id_tensor = torch.full((max_gid + 1,), -1, device=self.device, dtype=torch.long)
        valid_mask = torch.zeros((max_gid + 1,), device=self.device, dtype=torch.bool)
        group_index_tensor = torch.full((max_gid + 1,), -1, device=self.device, dtype=torch.long)

        for node in nodes_data:
            node_type, node_id = node["type"], node["id"]
            velocity_max = node.get("velocity_max", 0.0) or 0.0
            acceleration_max = node.get("acceleration_max", 0.0) or 0.0
            r_s = node.get("r_s", 0.0) or 0.0
            r_c = node.get("r_c", 0.0) or 0.0
            r_a = node.get("r_a", 0.0) or 0.0
            task = node.get("task", -1)
            if task is None:
                task = -1
            group_index = node.get("group_index", -1)
            if group_index is None:
                group_index = -1

            feat = torch.tensor(node["pos"], dtype=torch.float32, device=self.device)
            vel = torch.tensor(node["velocity"], dtype=torch.float32, device=self.device)
            extra = torch.tensor([velocity_max, acceleration_max, r_s, r_c, r_a], dtype=torch.float32, device=self.device)
            features = torch.cat([feat, vel, extra], dim=0)
            if features.numel() != feat_dim:
                raise ValueError(f"Feature dim mismatch for node {node_id}: {features.numel()} vs {feat_dim}")
            tid = self.type_name_to_id.get(node_type)
            if tid is None:
                raise ValueError(f"Unknown node type: {node_type}")

            state[node_id] = features
            task_tensor[node_id] = float(task)
            type_id_tensor[node_id] = int(tid)
            valid_mask[node_id] = True
            group_index_tensor[node_id] = int(group_index)

        return state, task_tensor, type_id_tensor, valid_mask, group_index_tensor

    def _load_nodes_from_json_batch(self, filepaths):
        base_state = self.state
        base_task = self.task_tensor
        base_type_id = self.type_id_tensor
        base_valid = self.valid_node_mask
        base_group_index = self.group_index_tensor

        state_list = [base_state]
        task_list = [base_task]
        for filepath in filepaths[1:]:
            with open(filepath, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)
            state, task_tensor, type_id_tensor, valid_mask, group_index_tensor = self._build_state_from_nodes_data(nodes_data)
            if state.shape != base_state.shape:
                raise ValueError("Batch initial states must share the same node count.")
            if not torch.equal(valid_mask, base_valid):
                raise ValueError("Batch initial states must share the same valid node mask.")
            if not torch.equal(type_id_tensor, base_type_id):
                raise ValueError("Batch initial states must share the same node types.")
            if not torch.allclose(task_tensor[valid_mask], base_task[valid_mask]):
                raise ValueError("Batch initial states must share the same task values.")
            if not torch.allclose(state[valid_mask, 4:], base_state[valid_mask, 4:], rtol=1e-5, atol=1e-6):
                raise ValueError("Batch initial states must share the same non-pos/vel features.")
            if not torch.equal(group_index_tensor, base_group_index):
                raise ValueError("Batch initial states must share the same group_index.")
            state_list.append(state)
            task_list.append(task_tensor)

        self.state_b = torch.stack(state_list, dim=0).contiguous()
        self.task_b = torch.stack(task_list, dim=0).contiguous()



def save_state_series(save_dir: str | None, scene_type_id: int, payload: Dict, label: str | None = None) -> None:
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{label}" if label else ""
    path = os.path.join(save_dir, f"state_scene{scene_type_id}{suffix}.pt")
    torch.save(payload, path)








def _amp_dtype_from_str(amp_dtype: str):
    s = (amp_dtype or "").lower().strip()
    if s in ("", "none", "off", "false", "0", "fp32", "float32"):
        return None
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype={amp_dtype!r}; use bf16/fp16/off")


def _f1_torch(f: torch.Tensor, f0: float, d: float) -> torch.Tensor:
    x0 = float(f0)
    dd = float(d)
    x = f
    left = x < (x0 - dd)
    mid = (x >= (x0 - dd)) & (x < x0)
    s_val = torch.zeros_like(x)
    s_val = torch.where(left, torch.ones_like(x), s_val)
    s_mid = 0.5 * (1.0 - torch.cos(math.pi * (x - x0) / max(dd, 1e-12)))
    s_val = torch.where(mid, s_mid, s_val)
    return 1.0 - s_val


def _f3_torch(f: torch.Tensor, a: float) -> torch.Tensor:
    aa = float(a)
    return (aa * aa) / ((f + aa) * (f + aa))

class NetStatic:
    def __init__(self, net: Net, device: torch.device):
        self.device = device
        self.net_constraint = net.net_constraint
        self.type_name_to_id = dict(net.type_name_to_id)
        self.id_to_type_name = dict(net.id_to_type_name)

        self.state0 = net.state.detach().to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        self.task0 = net.task_tensor.detach().to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        self.valid_node_mask = net.valid_node_mask.detach().to(device=device, dtype=torch.bool, non_blocking=True).contiguous()

        raw_type_id = net.type_id_tensor.detach().to(device=device, dtype=torch.long, non_blocking=True).contiguous()
        type_id = raw_type_id.clone()
        type_id[~self.valid_node_mask] = 0
        type_id[type_id < 0] = 0
        self.type_id = type_id

        self.N = int(self.state0.shape[0])
        self.F = int(self.state0.shape[1])
        self.type_count = int(len(self.type_name_to_id))

        group_index = getattr(net, "group_index_tensor", None)
        if group_index is None:
            self.group_index = torch.full_like(self.task0, -1, dtype=torch.long)
        else:
            self.group_index = group_index.detach().to(device=device, dtype=torch.long, non_blocking=True).contiguous()

        self.net_names = list(self.net_constraint.keys())
        self.net_name_to_id = {name: i for i, name in enumerate(self.net_names)}
        self.id_to_net_name = {i: name for name, i in self.net_name_to_id.items()}
        self.net_count = int(len(self.net_names))

        type_names = [None] * self.type_count
        for name, tid in self.type_name_to_id.items():
            if tid < 0 or tid >= self.type_count:
                continue
            type_names[tid] = name

        env_type_ids = [tid for tid, name in enumerate(type_names) if name and "environment" in name]
        self.env_type_ids_t = torch.tensor(env_type_ids, device=device, dtype=torch.long) if env_type_ids else None
        if self.env_type_ids_t is not None and self.env_type_ids_t.numel() > 0:
            self.all_env_mask = torch.isin(self.type_id, self.env_type_ids_t)
        else:
            self.all_env_mask = torch.zeros_like(self.type_id, dtype=torch.bool)
        self.non_env_mask = self.valid_node_mask & (~self.all_env_mask)
        self.non_env_count = int(self.non_env_mask.sum().item())

        self.allowed_type_mask = torch.zeros(
            (self.net_count, self.type_count, self.type_count),
            device=device,
            dtype=torch.bool,
        )
        for net_name in self.net_names:
            net_id = self.net_name_to_id[net_name]
            for edge in self.net_constraint.get(net_name, []):
                src, dst = edge.split("_")[:2]
                if (src not in self.type_name_to_id) or (dst not in self.type_name_to_id):
                    continue
                si = self.type_name_to_id[src]
                di = self.type_name_to_id[dst]
                self.allowed_type_mask[net_id, si, di] = True

        self.pair_index = (self.type_id[:, None] * self.type_count + self.type_id[None, :]).contiguous()
        self._allowed_pairs_nodes: List[torch.Tensor] = []

        for net_id, net_name in enumerate(self.net_names):
            allow = self.allowed_type_mask[net_id, self.type_id[:, None], self.type_id[None, :]]
            allow = allow.contiguous()
            self._allowed_pairs_nodes.append(allow)

        self.comm_net_id = self.net_name_to_id.get("communication_net")
        self.logic_net_id = self.net_name_to_id.get("logic_net")
        self.enemy_logic_net_id = self.net_name_to_id.get("enemy_logic_net")
        self.action_net_id = self.net_name_to_id.get("action_net")

        process_type_ids = []
        for name, tid in self.type_name_to_id.items():
            mask = (net.type_id_tensor == tid)
            if mask.any() and (net.task_tensor[mask] > 0).any():
                process_type_ids.append(tid)
        self.process_type_ids_t = (
            torch.tensor(process_type_ids, device=device, dtype=torch.long) if process_type_ids else None
        )

    def allowed_pairs_nodes(self, net_id: int) -> torch.Tensor:
        return self._allowed_pairs_nodes[net_id]


class ParamTablesBatch:
    def __init__(
        self,
        C_rep_table_b: torch.Tensor,
        C_align_table_b: torch.Tensor,
        C_attr_table_b: torch.Tensor,
        self_drive_target_b: torch.Tensor,
        self_drive_gain_b: torch.Tensor,
        enemy_drive_target_b: torch.Tensor,
        enemy_drive_gain_b: torch.Tensor,
    ):
        self.C_rep_table_b = C_rep_table_b
        self.C_align_table_b = C_align_table_b
        self.C_attr_table_b = C_attr_table_b
        self.self_drive_target_b = self_drive_target_b
        self.self_drive_gain_b = self_drive_gain_b
        self.enemy_drive_target_b = enemy_drive_target_b
        self.enemy_drive_gain_b = enemy_drive_gain_b

        B, net_count, type_count, _ = self.C_rep_table_b.shape
        L = type_count * type_count
        self._C_rep_flat = self.C_rep_table_b.reshape(B, net_count, L)
        self._C_align_flat = self.C_align_table_b.reshape(B, net_count, L)
        self._C_attr_flat = self.C_attr_table_b.reshape(B, net_count, L)

    def gather_pair_table(self, flat: torch.Tensor, net_id: int, pair_index_block: torch.Tensor) -> torch.Tensor:
        return flat[:, net_id, :][:, pair_index_block]


def build_param_tables_batch(static: NetStatic, parameters_list: List[Dict]) -> ParamTablesBatch:
    B = int(len(parameters_list))
    device = static.device
    net_count = static.net_count
    type_count = static.type_count

    def zeros():
        return torch.zeros((B, net_count, type_count, type_count), device=device, dtype=torch.float32)

    C_rep = zeros()
    C_align = zeros()
    C_attr = zeros()

    self_drive_target = torch.zeros((B, type_count), device=device, dtype=torch.float32)
    self_drive_gain = torch.zeros((B, type_count), device=device, dtype=torch.float32)
    enemy_drive_target = torch.zeros((B, type_count), device=device, dtype=torch.float32)
    enemy_drive_gain = torch.zeros((B, type_count), device=device, dtype=torch.float32)

    per_net_params: Dict[str, List[Dict]] = {}
    for net_name in static.net_names:
        per_net_params[net_name] = [p.get(net_name, {}) or {} for p in parameters_list]

    for net_name in static.net_names:
        net_id = static.net_name_to_id[net_name]
        net_params_b = per_net_params[net_name]
        rep_gain_b = [d.get("rep_gain", {}) or {} for d in net_params_b]
        align_gain_b = [d.get("align_gain", {}) or {} for d in net_params_b]
        attr_gain_b = [d.get("attraction_gain", {}) or {} for d in net_params_b]

        for edge in static.net_constraint.get(net_name, []):
            src, dst = edge.split("_")[:2]
            if (src not in static.type_name_to_id) or (dst not in static.type_name_to_id):
                continue
            si = static.type_name_to_id[src]
            di = static.type_name_to_id[dst]
            key = f"{src}_{dst}"

            C_rep_v = torch.tensor([float(rep_gain_b[i].get(key, 0.0)) for i in range(B)], device=device)
            C_align_v = torch.tensor([float(align_gain_b[i].get(key, 0.0)) for i in range(B)], device=device)
            C_attr_v = torch.tensor([float(attr_gain_b[i].get(key, 0.0)) for i in range(B)], device=device)

            C_rep[:, net_id, si, di] = C_rep_v
            C_align[:, net_id, si, di] = C_align_v
            C_attr[:, net_id, si, di] = C_attr_v

    for b, p in enumerate(parameters_list):
        sd = p.get("self_drive")
        if isinstance(sd, dict):
            target_v = sd.get("self_drive_velocity", {}) or {}
            gain_v = sd.get("self_drive_gain", {}) or {}
            for k, v in target_v.items():
                tid = static.type_name_to_id.get(k) if isinstance(k, str) else int(k)
                if tid is None or tid < 0 or tid >= type_count:
                    continue
                self_drive_target[b, tid] = float(v)
            for k, v in gain_v.items():
                tid = static.type_name_to_id.get(k) if isinstance(k, str) else int(k)
                if tid is None or tid < 0 or tid >= type_count:
                    continue
                self_drive_gain[b, tid] = float(v)

        enemy_sd = p.get("enemy_self_drive")
        if isinstance(enemy_sd, dict):
            target_v = enemy_sd.get("self_drive_velocity", {}) or {}
            gain_v = enemy_sd.get("self_drive_gain", {}) or {}
            for k, v in target_v.items():
                tid = static.type_name_to_id.get(k) if isinstance(k, str) else int(k)
                if tid is None or tid < 0 or tid >= type_count:
                    continue
                enemy_drive_target[b, tid] = float(v)
            for k, v in gain_v.items():
                tid = static.type_name_to_id.get(k) if isinstance(k, str) else int(k)
                if tid is None or tid < 0 or tid >= type_count:
                    continue
                enemy_drive_gain[b, tid] = float(v)

    return ParamTablesBatch(
        C_rep_table_b=C_rep.contiguous(),
        C_align_table_b=C_align.contiguous(),
        C_attr_table_b=C_attr.contiguous(),
        self_drive_target_b=self_drive_target.contiguous(),
        self_drive_gain_b=self_drive_gain.contiguous(),
        enemy_drive_target_b=enemy_drive_target.contiguous(),
        enemy_drive_gain_b=enemy_drive_gain.contiguous(),
    )


class NetBatch:
    def __init__(self, static: NetStatic, batch_size: int, state_init_b: torch.Tensor | None = None, task_init_b: torch.Tensor | None = None):
        self.static = static
        self.B = int(batch_size)
        if state_init_b is not None:
            if int(state_init_b.shape[0]) != self.B:
                raise ValueError("state_init_b batch size mismatch.")
            self.state_b = state_init_b.to(device=static.device, dtype=torch.float32, non_blocking=True).contiguous()
        else:
            self.state_b = static.state0.unsqueeze(0).repeat(self.B, 1, 1).contiguous()
        if task_init_b is not None:
            if int(task_init_b.shape[0]) != self.B:
                raise ValueError("task_init_b batch size mismatch.")
            self.task_b = task_init_b.to(device=static.device, dtype=torch.float32, non_blocking=True).contiguous()
        else:
            self.task_b = static.task0.unsqueeze(0).repeat(self.B, 1).contiguous()

        self.valid_node_mask = static.valid_node_mask
        self.all_env_mask = static.all_env_mask
        self.type_id = static.type_id

        self.dist_full_b: torch.Tensor | None = None
        self.adj_by_net: List[torch.Tensor] = [
            torch.zeros((self.B, static.N, static.N), device=static.device, dtype=torch.bool) for _ in range(static.net_count)
        ]


def _apply_topk_src(adj: torch.Tensor, dist: torch.Tensor, k_src: torch.Tensor) -> torch.Tensor:
    if adj is None or adj.numel() == 0 or not adj.any():
        return adj
    if k_src.numel() == 0:
        return torch.zeros_like(adj, dtype=torch.bool)
    k_max = int(k_src.max().item())
    if k_max <= 0:
        return torch.zeros_like(adj, dtype=torch.bool)
    k_max = min(k_max, int(adj.shape[2]))
    dist_masked = dist.masked_fill(~adj, float("inf"))
    values, indices = torch.topk(dist_masked, k=k_max, dim=2, largest=False)
    rank = torch.arange(k_max, device=adj.device)[None, None, :]
    keep = rank < k_src[:, :, None]
    valid = keep & torch.isfinite(values)
    out = torch.zeros_like(adj, dtype=torch.float32)
    out.scatter_(2, indices, valid.to(torch.float32))
    return out > 0.0


def net_build_batched(static: NetStatic, nb: NetBatch, tables: ParamTablesBatch) -> None:
    B, N = nb.B, static.N
    device = static.device

    pos_b = nb.state_b[:, :, IDX_POS].to(dtype=torch.float32)
    dist = torch.cdist(pos_b, pos_b, p=2)
    dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))

    active_mask_b = static.valid_node_mask[None, :] & (nb.task_b != 0)
    dist = dist.masked_fill(~active_mask_b[:, :, None], float("inf"))
    dist = dist.masked_fill(~active_mask_b[:, None, :], float("inf"))
    nb.dist_full_b = dist.contiguous()

    for net_id in range(static.net_count):
        nb.adj_by_net[net_id].fill_(False)

    r_s = nb.state_b[:, :, IDX_R_S]
    r_c = nb.state_b[:, :, IDX_R_C]
    r_a = nb.state_b[:, :, IDX_R_A]

    for net_name in static.net_names:
        if "action_net" in net_name or "logic_net" in net_name:
            continue
        net_id = static.net_name_to_id[net_name]
        if "communication_net" in net_name:
            allow = static.allowed_pairs_nodes(net_id)
            threshold = r_c[:, :, None]
            adj = allow[None, :, :] & (dist <= threshold) & torch.isfinite(dist)
        elif "sensing_net" in net_name:
            allow = static.allowed_pairs_nodes(net_id)
            threshold = r_s[:, None, :]
            adj = allow[None, :, :] & (dist <= threshold) & torch.isfinite(dist)
        else:
            allow = static.allowed_pairs_nodes(net_id)
            threshold = torch.full((B, N, 1), 100.0, device=device, dtype=dist.dtype)
            adj = allow[None, :, :] & (dist <= threshold) & torch.isfinite(dist)
        adj = adj & (active_mask_b[:, :, None] & active_mask_b[:, None, :])
        nb.adj_by_net[net_id].copy_(adj)

    k_val = min(10, N)
    if k_val > 0:
        k_src = torch.full((B, N), k_val, device=device, dtype=torch.long)

        # Relay source nodes are unlimited in the baseline and capped only for
        # the RelayTopK ablation.  All non-relay communication remains top-k=10.
        relay_tids = [
            tid for name, tid in static.type_name_to_id.items() if name is not None and "relay" in name
        ]
        if relay_tids:
            dec_mask = torch.isin(static.type_id, torch.tensor(relay_tids, device=device))
            relay_k = min(RELAY_TOPK, N)
            k_src[:, dec_mask] = relay_k

        for comm_net_name in ("communication_net", "enemy_communication_net"):
            comm_id = static.net_name_to_id.get(comm_net_name)
            if comm_id is None:
                continue
            adj = nb.adj_by_net[comm_id]
            if not adj.any():
                continue
            adj = _apply_topk_src(adj, dist, k_src)
            nb.adj_by_net[comm_id].copy_(adj)

    logic_pairs = (
        ("logic_net", "communication_net", "sensing_net"),
        ("enemy_logic_net", "enemy_communication_net", "enemy_sensing_net"),
    )
    for logic_name, comm_name, detect_name in logic_pairs:
        logic_id = static.net_name_to_id.get(logic_name)
        if logic_id is None:
            continue
        logic_adj = torch.zeros((B, N, N), device=device, dtype=torch.bool)
        comm_id = static.net_name_to_id.get(comm_name)
        det_id = static.net_name_to_id.get(detect_name)
        if comm_id is not None:
            logic_adj |= nb.adj_by_net[comm_id]
        if det_id is not None:
            logic_adj |= nb.adj_by_net[det_id]
        allow = static.allowed_pairs_nodes(logic_id)
        logic_adj = logic_adj & allow[None, :, :]
        logic_adj = logic_adj & (active_mask_b[:, :, None] & active_mask_b[:, None, :])
        nb.adj_by_net[logic_id].copy_(logic_adj)

    
    for detect_net_name, logic_name, relay_type in (
        ("sensing_net", "logic_net", "relay"),
        ("enemy_sensing_net", "enemy_logic_net", "enemyrelay"),
    ):
        detect_id = static.net_name_to_id.get(detect_net_name)
        logic_id = static.net_name_to_id.get(logic_name)
        if detect_id is None or logic_id is None:
            continue

        base_adj = nb.adj_by_net[detect_id]
        logic_adj = nb.adj_by_net[logic_id]
        if not logic_adj.any():
            continue

        reach2 = torch.bmm(logic_adj.to(torch.float16), logic_adj.to(torch.float16)) > 0.0

        dec_tid = static.type_name_to_id.get(relay_type)
        if dec_tid is None:
            continue
        relay_mask = static.type_id == int(dec_tid)
        env_mask = static.all_env_mask

        supplement = reach2 & env_mask[None, :, None] & relay_mask[None, None, :]
        allow = static.allowed_pairs_nodes(detect_id)
        final = (base_adj | supplement) & allow[None, :, :]
        final = final & (active_mask_b[:, :, None] & active_mask_b[:, None, :])
        nb.adj_by_net[detect_id].copy_(final)

    for action_net_name in ("action_net", "enemy_action_net"):
        action_id = static.net_name_to_id.get(action_net_name)
        if action_id is None:
            continue

        if "enemy" in action_net_name:
            logic_name = "enemy_logic_net"
            action_type_name = "enemyaction"
        else:
            logic_name = "logic_net"
            action_type_name = "action"

        logic_id = static.net_name_to_id.get(logic_name)
        logic_adj = nb.adj_by_net[logic_id] if logic_id is not None else torch.zeros((B, N, N), device=device, dtype=torch.bool)

        logic_sym = logic_adj | logic_adj.transpose(1, 2)
        if logic_sym.any():
            reach2 = torch.bmm(logic_sym.to(torch.float16), logic_sym.to(torch.float16)) > 0.0
        else:
            reach2 = torch.zeros((B, N, N), device=device, dtype=torch.bool)
        if reach2.any():
            reach3 = torch.bmm(reach2.to(torch.float16), logic_sym.to(torch.float16)) > 0.0
        else:
            reach3 = torch.zeros((B, N, N), device=device, dtype=torch.bool)

        common = logic_sym | reach2 | reach3
        if not common.any():
            nb.adj_by_net[action_id].fill_(False)
            continue

        allow = static.allowed_pairs_nodes(action_id)
        action_tid = static.type_name_to_id.get(action_type_name)
        if action_tid is None:
            threshold = torch.full((B, N), 100.0, device=device, dtype=dist.dtype)
        else:
            src_is_action = static.type_id == int(action_tid)
            thr_src = torch.where(r_a == 0, torch.full_like(r_a, 100.0), r_a)
            threshold = torch.where(src_is_action[None, :], thr_src, torch.full_like(thr_src, 100.0))

        adj = common & allow[None, :, :] & (dist <= threshold[:, :, None]) & torch.isfinite(dist)
        adj = adj & (active_mask_b[:, :, None] & active_mask_b[:, None, :])
        nb.adj_by_net[action_id].copy_(adj)

class ForceDynamicsBatch:
    def __init__(self, static: NetStatic, dt: float, scene_type_id: int, tables: ParamTablesBatch):
        self.static = static
        self.dt = float(dt)
        self.scene_type_id = int(scene_type_id)
        self.tables = tables

    def compute_forces(self, nb: NetBatch, force_record: Dict | None = None) -> torch.Tensor:
        B, N = nb.B, self.static.N
        device = self.static.device

        state = nb.state_b
        pos = state[:, :, IDX_POS].to(dtype=torch.float32)
        vel = state[:, :, IDX_VEL].to(dtype=torch.float32)

        forces = torch.zeros((B, N, 2), device=device, dtype=torch.float32)

        pair_index = self.static.pair_index

        for net_id in range(self.static.net_count):
            net_name = self.static.net_names[net_id] if net_id < len(self.static.net_names) else f"net{net_id}"
            
            if net_name in ("logic_net", "enemy_logic_net"):
                continue

            adj = nb.adj_by_net[net_id]
            if not adj.any():
                continue

            w = adj.to(torch.float32)
            if not (w > 0).any():
                continue

            for j0 in range(0, N, 256):
                j1 = min(j0 + 256, N)
                pair_index_block = pair_index[:, j0:j1]
                w_block = w[:, :, j0:j1]
                mask = w_block > 0
                if not mask.any():
                    continue

                C_rep = self.tables.gather_pair_table(self.tables._C_rep_flat, net_id, pair_index_block)
                C_align = self.tables.gather_pair_table(self.tables._C_align_flat, net_id, pair_index_block)
                C_attr = self.tables.gather_pair_table(self.tables._C_attr_flat, net_id, pair_index_block)

                zero = torch.zeros((), device=device, dtype=torch.float32)
                C_rep = torch.where(mask, C_rep, zero)
                C_align = torch.where(mask, C_align, zero)
                C_attr = torch.where(mask, C_attr, zero)

                pos_src = pos
                pos_dst = pos[:, j0:j1, :]
                vel_src = vel
                vel_dst = vel[:, j0:j1, :]

                pos_vec = pos_dst[:, None, :, :] - pos_src[:, :, None, :]
                dist_sq = (pos_vec * pos_vec).sum(dim=-1, keepdim=True) + 1e-6
                dist_cube = dist_sq * torch.sqrt(dist_sq)

                f_rep = C_rep[:, :, :, None] * pos_vec / dist_cube
                f_align = C_align[:, :, :, None] * (vel_src[:, :, None, :] - vel_dst[:, None, :, :]) / torch.sqrt(1.0 + dist_sq)
                f_attr = C_attr[:, :, :, None] * (pos_src[:, :, None, :] - pos_dst[:, None, :, :])

                f = (f_rep + f_align + f_attr) * w_block[:, :, :, None]
                if force_record is not None:
                    per_net = (force_record.get("per_net") or {}).get(net_name)
                    total_mat = force_record.get("total")
                    comp_mat = None
                    total_comp = None
                    if force_record.get("per_net_components") is not None:
                        comp_mat = force_record["per_net_components"].get(net_name)
                    if force_record.get("total_components") is not None:
                        total_comp = force_record["total_components"]
                    if per_net is not None:
                        per_net[:, j0:j1, :, :] += f.permute(0, 2, 1, 3)
                        if total_mat is not None:
                            total_mat[:, j0:j1, :, :] += f.permute(0, 2, 1, 3)
                        if comp_mat is not None:
                            comp_mat["rep"][:, j0:j1, :, :] += f_rep.permute(0, 2, 1, 3)
                            comp_mat["align"][:, j0:j1, :, :] += f_align.permute(0, 2, 1, 3)
                            comp_mat["attr"][:, j0:j1, :, :] += f_attr.permute(0, 2, 1, 3)
                            if total_comp is not None:
                                total_comp["rep"][:, j0:j1, :, :] += f_rep.permute(0, 2, 1, 3)
                                total_comp["align"][:, j0:j1, :, :] += f_align.permute(0, 2, 1, 3)
                                total_comp["attr"][:, j0:j1, :, :] += f_attr.permute(0, 2, 1, 3)
                forces[:, j0:j1, :] += f.sum(dim=1)

        if (self.tables.self_drive_gain_b.abs().sum() > 0) or (self.tables.self_drive_target_b.abs().sum() > 0):
            vel_norm = torch.norm(vel, dim=2, keepdim=True) + 1e-8
            type_id = self.static.type_id
            target = self.tables.self_drive_target_b[:, type_id].unsqueeze(2)
            gain = self.tables.self_drive_gain_b[:, type_id].unsqueeze(2)
            drive_scalar = (target - vel_norm) / vel_norm
            if self.scene_type_id == 1:
                drive_direction = torch.tensor(
                    SELF_DRIVE_DIRECTION, device=device, dtype=torch.float32,
                ).view(1, 1, 2)
            else:
                drive_direction = vel / vel_norm
            drive = gain * drive_scalar * drive_direction
            forces = forces + drive
            if force_record is not None:
                if force_record.get("self_drive") is not None:
                    force_record["self_drive"] += drive
                if force_record.get("total") is not None:
                    idx = torch.arange(self.static.N, device=device)
                    force_record["total"][:, idx, idx, :] += drive

        if (self.tables.enemy_drive_gain_b.abs().sum() > 0) or (self.tables.enemy_drive_target_b.abs().sum() > 0):
            vel_norm = torch.norm(vel, dim=2, keepdim=True) + 1e-8
            vel_dir = vel / vel_norm
            type_id = self.static.type_id
            target = self.tables.enemy_drive_target_b[:, type_id].unsqueeze(2)
            gain = self.tables.enemy_drive_gain_b[:, type_id].unsqueeze(2)
            drive_scalar = (target - vel_norm) / vel_norm
            drive = gain * drive_scalar * vel_dir
            forces = forces + drive
            if force_record is not None:
                if force_record.get("self_drive") is not None:
                    force_record["self_drive"] += drive
                if force_record.get("total") is not None:
                    idx = torch.arange(self.static.N, device=device)
                    force_record["total"][:, idx, idx, :] += drive

        alpha = 10.0
        beta = 0.001
        non_env = ~self.static.all_env_mask
        if non_env.any():
            node_force = forces[:, non_env, :]
            mag = torch.norm(node_force, dim=2, keepdim=True)
            normed = alpha * torch.tanh(beta * mag)
            forces[:, non_env, :] = normed * (node_force / (mag + 1e-6))
        return forces

    def step(self, nb: NetBatch, force_record: Dict | None = None) -> torch.Tensor:
        forces = self.compute_forces(nb, force_record=force_record)
        dt = self.dt

        task = nb.task_b
        pos = nb.state_b[:, :, IDX_POS]
        vel = nb.state_b[:, :, IDX_VEL]

        active_non_env = (task != 0) & (~self.static.all_env_mask[None, :])

        acc = forces
        acc_max = nb.state_b[:, :, IDX_AMAX].view(nb.B, self.static.N, 1)
        acc_norm = torch.norm(acc, dim=2, keepdim=True)
        acc = torch.where(acc_norm > acc_max, acc * (acc_max / (acc_norm + 1e-6)), acc)

        vel_new = vel + acc * dt
        v_max = nb.state_b[:, :, IDX_VMAX].view(nb.B, self.static.N, 1)
        v_norm = torch.norm(vel_new, dim=2, keepdim=True)
        vel_new = torch.where(v_norm > v_max, vel_new * (v_max / (v_norm + 1e-6)), vel_new)

        pos_new = pos + vel_new * dt

        am = active_non_env[:, :, None]
        
        nb.state_b[:, :, IDX_VEL] = torch.where(am, vel_new, vel)
        nb.state_b[:, :, IDX_POS] = torch.where(am, pos_new, pos)
        return forces

    def update_task_info(self, nb: NetBatch) -> None:
        process_type_ids = self.static.process_type_ids_t
        if process_type_ids is None or process_type_ids.numel() == 0:
            return
        dst_process_mask = torch.isin(self.static.type_id, process_type_ids)
        if not dst_process_mask.any():
            return

        pos = nb.state_b[:, :, IDX_POS].to(torch.float32)
        dist = torch.cdist(pos, pos, p=2)
        dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))

        for action_net_name, action_type_name in (("action_net", "action"), ("enemy_action_net", "enemyaction")):
            net_id = self.static.net_name_to_id.get(action_net_name)
            if net_id is None:
                continue
            adj = nb.adj_by_net[net_id]
            if not adj.any():
                continue

            action_tid = self.static.type_name_to_id.get(action_type_name)
            if action_tid is None:
                continue
            src_action_mask = self.static.type_id == int(action_tid)
            if not src_action_mask.any():
                continue

            valid_edges = adj & src_action_mask[None, :, None] & dst_process_mask[None, None, :]
            valid_edges = valid_edges & (dist < 10.0)
            if not valid_edges.any():
                continue

            masked_dist = torch.where(valid_edges, dist, torch.full_like(dist, float("inf")))
            min_dist, argmin = masked_dist.min(dim=2)
            has_choice = torch.isfinite(min_dist) & src_action_mask[None, :]
            if not has_choice.any():
                continue

            b_idx, src_idx = has_choice.nonzero(as_tuple=True)
            dst_idx = argmin[b_idx, src_idx]
            if dst_idx.numel() == 0:
                continue

            impact = torch.full((dst_idx.numel(),), float(self.dt), device=self.static.device, dtype=torch.float32)
            BN = nb.B * self.static.N
            task_flat = nb.task_b.view(BN)
            dst_flat = b_idx * self.static.N + dst_idx
            task_flat.index_add_(0, dst_flat, -impact)
            unique_dst = torch.unique(dst_flat)
            task_flat[unique_dst] = torch.clamp(task_flat[unique_dst], min=0.0)

class PerformanceEvaluationBatch:
    def __init__(self, static: NetStatic):
        self.static = static

    def phi_collision_avoidance(self, nb: NetBatch, r_coll: float = 3.0) -> torch.Tensor:
        if self.static.non_env_count < 2:
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)

        pos = nb.state_b[:, :, IDX_POS].to(torch.float32)
        task = nb.task_b
        mask_all = task != 0

        dist = torch.cdist(pos, pos, p=2)
        dist.diagonal(dim1=1, dim2=2).fill_(0.0)
        dist = dist.masked_fill((~mask_all)[:, :, None], float("inf"))
        dist = dist.masked_fill((~mask_all)[:, None, :], float("inf"))

        coll = (dist < float(r_coll)) & (dist > 0)
        total = coll.sum(dim=(1, 2), dtype=torch.float32) * 0.5

        env_mask_all = self.static.all_env_mask[None, :] & mask_all
        env_pairs = env_mask_all[:, :, None] & env_mask_all[:, None, :]
        env_coll = (coll & env_pairs).sum(dim=(1, 2), dtype=torch.float32) * 0.5

        total = torch.clamp(total - env_coll, min=0.0)
        denom = float(self.static.non_env_count * (self.static.non_env_count - 1))
        if denom <= 0:
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)
        return total / denom

    def phi_correlation(self, nb: NetBatch, comm_net_id: int | None) -> torch.Tensor:
        if comm_net_id is None:
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)
        adj = nb.adj_by_net[comm_net_id]
        if not adj.any():
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)

        non_env_mask = self.static.non_env_mask
        if not non_env_mask.any():
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)
        idx = non_env_mask.nonzero(as_tuple=True)[0]

        vel = nb.state_b[:, idx, IDX_VEL].to(torch.float32)
        vel_norm = vel / (torch.norm(vel, dim=2, keepdim=True) + 1e-8)
        sim = torch.bmm(vel_norm, vel_norm.transpose(1, 2))
        M = vel.shape[1]
        eye = torch.eye(M, device=self.static.device, dtype=torch.bool)
        sim = sim.masked_fill(eye[None, :, :], 0.0)

        adj_sub = adj[:, idx][:, :, idx]
        final = sim * adj_sub.to(sim.dtype)
        row_sums = final.sum(dim=1)
        valid_neighbors = adj_sub.sum(dim=1, dtype=torch.float32)
        valid_counts = torch.where(valid_neighbors > 0, valid_neighbors, torch.ones_like(valid_neighbors))
        row_means = row_sums / valid_counts
        num_agents = float(M)
        return row_means.sum(dim=1) / max(num_agents, 1.0)

    @staticmethod
    def _union_find_flat(num_nodes: int, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if src.numel() == 0:
            return torch.arange(num_nodes, device=src.device)
        parent = torch.arange(num_nodes, device=src.device)
        for _ in range(15):
            parent = parent[parent]
            ru = parent[src]
            rv = parent[dst]
            m = ru != rv
            if not m.any():
                break
            au = ru[m]
            av = rv[m]
            mn = torch.minimum(au, av)
            mx = torch.maximum(au, av)
            parent.scatter_reduce_(0, mx, mn, reduce="amin", include_self=True)
        for _ in range(3):
            parent = parent[parent]
        return parent

    def phi_gather_crossing(
        self,
        nb: NetBatch,
        comm_net_id: int | None,
        min_node_count: int | None = None,
    ) -> torch.Tensor:
        total_nodes = self.static.non_env_count
        if total_nodes <= 0:
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)
        target_count = max(1, int(min_node_count)) if min_node_count is not None else max(1, int(total_nodes))

        combined = torch.zeros((nb.B, self.static.N, self.static.N), device=self.static.device, dtype=torch.bool)
        if comm_net_id is not None:
            combined |= nb.adj_by_net[comm_net_id]

        combined = combined | combined.transpose(1, 2)
        combined = combined & self.static.non_env_mask[None, :, None] & self.static.non_env_mask[None, None, :]

        if not combined.any():
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)

        b_idx, src, dst = combined.nonzero(as_tuple=True)
        src_g = b_idx * self.static.N + src
        dst_g = b_idx * self.static.N + dst
        labels = self._union_find_flat(nb.B * self.static.N, src_g, dst_g)

        active_mask = self.static.non_env_mask[None, :].expand(nb.B, self.static.N)
        active_nodes = active_mask.nonzero(as_tuple=False)
        act_b = active_nodes[:, 0]
        act_n = active_nodes[:, 1]
        act_g = act_b * self.static.N + act_n
        roots = labels[act_g]
        counts = torch.bincount(roots, minlength=nb.B * self.static.N).view(nb.B, self.static.N)

        max_size = counts.to(torch.float32).max(dim=1).values
        ratio = max_size / float(target_count)
        return torch.clamp(ratio, max=1.0)

    def phi_gather_searchandrescue(
        self,
        nb: NetBatch,
        comm_net_id: int | None,
        min_node_count: int = 1,
    ) -> torch.Tensor:
        total_nodes = self.static.non_env_count
        if total_nodes <= 0:
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)

        combined = torch.zeros((nb.B, self.static.N, self.static.N), device=self.static.device, dtype=torch.bool)
        if comm_net_id is not None:
            combined |= nb.adj_by_net[comm_net_id]

        combined = combined | combined.transpose(1, 2)
        combined = combined & self.static.non_env_mask[None, :, None] & self.static.non_env_mask[None, None, :]

        if not combined.any():
            return torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)

        b_idx, src, dst = combined.nonzero(as_tuple=True)
        src_g = b_idx * self.static.N + src
        dst_g = b_idx * self.static.N + dst
        labels = self._union_find_flat(nb.B * self.static.N, src_g, dst_g)

        active_mask = self.static.non_env_mask[None, :].expand(nb.B, self.static.N)
        active_nodes = active_mask.nonzero(as_tuple=False)
        act_b = active_nodes[:, 0]
        act_n = active_nodes[:, 1]
        act_g = act_b * self.static.N + act_n
        roots = labels[act_g]
        counts = torch.bincount(roots, minlength=nb.B * self.static.N).view(nb.B, self.static.N)

        counts_f = counts.to(torch.float32)
        inf_mask = torch.full_like(counts_f, float("inf"))
        masked = torch.where(counts_f > 0, counts_f, inf_mask)
        min_size = masked.min(dim=1).values
        min_size = torch.where(torch.isfinite(min_size), min_size, torch.zeros_like(min_size))
        ratio = (min_size / float(max(min_node_count, 1))) ** 3
        return torch.clamp(ratio, max=1.0)

    def connectivity_component_sizes(
        self,
        nb: NetBatch,
        comm_net_id: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_nodes = self.static.non_env_count
        zeros = torch.zeros((nb.B,), device=self.static.device, dtype=torch.float32)
        if total_nodes <= 0:
            return zeros, zeros, zeros

        combined = torch.zeros((nb.B, self.static.N, self.static.N), device=self.static.device, dtype=torch.bool)
        if comm_net_id is not None:
            combined |= nb.adj_by_net[comm_net_id]
        combined = combined | combined.transpose(1, 2)
        combined = combined & self.static.non_env_mask[None, :, None] & self.static.non_env_mask[None, None, :]

        if combined.any():
            b_idx, src, dst = combined.nonzero(as_tuple=True)
            src_g = b_idx * self.static.N + src
            dst_g = b_idx * self.static.N + dst
            labels = self._union_find_flat(nb.B * self.static.N, src_g, dst_g)
        else:
            labels = torch.arange(nb.B * self.static.N, device=self.static.device)

        active_mask = self.static.non_env_mask[None, :].expand(nb.B, self.static.N)
        active_nodes = active_mask.nonzero(as_tuple=False)
        act_b = active_nodes[:, 0]
        act_n = active_nodes[:, 1]
        act_g = act_b * self.static.N + act_n
        roots = labels[act_g]
        counts = torch.bincount(roots, minlength=nb.B * self.static.N).view(nb.B, self.static.N)
        counts_f = counts.to(torch.float32)
        max_size = counts_f.max(dim=1).values
        inf_mask = torch.full_like(counts_f, float("inf"))
        masked = torch.where(counts_f > 0, counts_f, inf_mask)
        min_size = masked.min(dim=1).values
        min_size = torch.where(torch.isfinite(min_size), min_size, torch.zeros_like(min_size))
        component_count = (counts > 0).sum(dim=1, dtype=torch.float32)
        return max_size, min_size, component_count

    def phi_gather(self, nb: NetBatch, comm_net_id: int | None) -> torch.Tensor:
        return self.phi_gather_crossing(nb, comm_net_id)

    def phi_task_searchandrescue(self, nb: NetBatch, task_info: Dict) -> torch.Tensor:
        B = nb.B
        pos = nb.state_b[:, :, IDX_POS].to(torch.float32)
        range_area = task_info.get("range", 0.0)
        if torch.is_tensor(range_area):
            range_area_b = range_area.to(self.static.device, dtype=torch.float32)
        else:
            range_area_b = torch.full((B,), float(range_area), device=self.static.device, dtype=torch.float32)
        center = torch.tensor([0.0, 0.0], device=self.static.device, dtype=torch.float32)
        d = torch.norm(pos - center[None, None, :], dim=2)
        count_in_range = ((d < range_area_b[:, None]) & self.static.non_env_mask[None, :]).sum(dim=1, dtype=torch.float32)

        task_type_id = self.static.type_name_to_id.get("environmentTask")
        if task_type_id is None:
            return torch.zeros((B,), device=self.static.device, dtype=torch.float32)
        task_mask = (self.static.type_id == int(task_type_id)) & self.static.valid_node_mask
        task_values = nb.task_b[:, task_mask]
        total_nodes = task_values.shape[1]
        if total_nodes == 0:
            return torch.zeros((B,), device=self.static.device, dtype=torch.float32)
        completed = (task_values == 0).sum(dim=1, dtype=torch.float32)

        node_count = task_info.get("node_count", total_nodes)
        if torch.is_tensor(node_count):
            node_count_b = node_count.to(self.static.device, dtype=torch.float32)
        else:
            node_count_b = torch.full((B,), float(node_count), device=self.static.device, dtype=torch.float32)
        coefficient = count_in_range / node_count_b.clamp_min(1.0)
        return completed * coefficient / float(total_nodes)

    def phi_task_crossing(self, nb: NetBatch, avg_x_b: torch.Tensor, task_info: Dict, t: float) -> torch.Tensor:
        B, N = nb.B, self.static.N
        moving = (self.static.valid_node_mask & self.static.non_env_mask)[None, :]
        pos = nb.state_b[:, :, IDX_POS].to(torch.float32)
        x = pos[:, :, 0]
        y = pos[:, :, 1]

        x_list = task_info.get("x", None)
        y_upper_list = task_info.get("y_upper", None)
        y_lower_list = task_info.get("y_lower", None)
        if x_list is None:
            x_list = []
        if y_upper_list is None:
            y_upper_list = []
        if y_lower_list is None:
            y_lower_list = []
        if torch.is_tensor(x_list):
            x_values = x_list
            y_upper_values = y_upper_list
            y_lower_values = y_lower_list
            lengths = task_info.get("lengths")
            if lengths is None:
                lengths = torch.full((B,), x_values.shape[1], device=self.static.device, dtype=torch.long)
            x_min = task_info.get("x_min")
            x_max = task_info.get("x_max")
            if x_min is None or x_max is None:
                x_min = x_values[:, 0]
                x_max = x_values[:, -1]

            valid_seq = lengths >= 2
            if not valid_seq.any():
                return torch.zeros((B,), device=self.static.device, dtype=torch.float32)

            max_index = torch.clamp(lengths - 1, min=1, max=x_values.shape[1] - 1)
            x0 = torch.zeros_like(x)
            x1 = torch.zeros_like(x)
            u0 = torch.zeros_like(x)
            u1 = torch.zeros_like(x)
            l0 = torch.zeros_like(x)
            l1 = torch.zeros_like(x)
            for b in range(B):
                if not bool(valid_seq[b]):
                    continue
                max_idx = int(max_index[b].item())
                x_vals_b = x_values[b, : max_idx + 1].contiguous()
                u_vals_b = y_upper_values[b, : max_idx + 1].contiguous()
                l_vals_b = y_lower_values[b, : max_idx + 1].contiguous()
                x_b = x[b].contiguous()
                inds_b = torch.searchsorted(x_vals_b, x_b, right=True)
                inds_b = torch.clamp(inds_b, 1, max_idx)
                idx0 = inds_b - 1
                idx1 = inds_b
                x0[b] = torch.gather(x_vals_b, 0, idx0)
                x1[b] = torch.gather(x_vals_b, 0, idx1)
                u0[b] = torch.gather(u_vals_b, 0, idx0)
                u1[b] = torch.gather(u_vals_b, 0, idx1)
                l0[b] = torch.gather(l_vals_b, 0, idx0)
                l1[b] = torch.gather(l_vals_b, 0, idx1)
            slope_u = (u1 - u0) / (x1 - x0 + 1e-9)
            slope_l = (l1 - l0) / (x1 - x0 + 1e-9)
            y_u = u0 + slope_u * (x - x0)
            y_l = l0 + slope_l * (x - x0)

            above_max = moving & (x > x_max[:, None]) & valid_seq[:, None]
            valid_x = moving & (x >= x_min[:, None]) & (x <= x_max[:, None]) & valid_seq[:, None]
        else:
            if (not x_list) or (not y_upper_list) or (not y_lower_list):
                return torch.zeros((B,), device=self.static.device, dtype=torch.float32)

            x_values = torch.tensor(x_list, device=self.static.device, dtype=torch.float32)
            y_upper_values = torch.tensor(y_upper_list, device=self.static.device, dtype=torch.float32)
            y_lower_values = torch.tensor(y_lower_list, device=self.static.device, dtype=torch.float32)

            sorted_x, idx = torch.sort(x_values)
            sorted_upper = y_upper_values[idx]
            sorted_lower = y_lower_values[idx]

            x_min = sorted_x.min()
            x_max = sorted_x.max()

            above_max = moving & (x > x_max)
            valid_x = moving & (x >= x_min) & (x <= x_max)

            xf = x.contiguous().view(-1)
            inds = torch.searchsorted(sorted_x, xf, right=True)
            inds = torch.clamp(inds, 1, sorted_x.numel() - 1)
            x0 = sorted_x[inds - 1]
            x1 = sorted_x[inds]
            u0 = sorted_upper[inds - 1]
            u1 = sorted_upper[inds]
            l0 = sorted_lower[inds - 1]
            l1 = sorted_lower[inds]
            slope_u = (u1 - u0) / (x1 - x0 + 1e-9)
            slope_l = (l1 - l0) / (x1 - x0 + 1e-9)
            y_u = (u0 + slope_u * (xf - x0)).reshape(B, N)
            y_l = (l0 + slope_l * (xf - x0)).reshape(B, N)

        in_channel = valid_x & (y >= y_l) & (y <= y_u)
        total_nodes = moving.sum(dim=1, dtype=torch.float32)
        nodes_in_channel = in_channel.sum(dim=1, dtype=torch.float32)
        nodes_task_succeed = above_max.sum(dim=1, dtype=torch.float32)
        nodes_out = (total_nodes - nodes_in_channel - nodes_task_succeed).clamp_min(0.0)
        nodes_fail = nodes_out

        sum_x = (x * moving.to(x.dtype)).sum(dim=1)
        mean_x = sum_x / total_nodes.clamp_min(1.0)
        coef = (mean_x - avg_x_b) / (10.0 * float(t) + 1e-6)
        phi = coef * (nodes_in_channel + nodes_task_succeed) / (nodes_fail * 100.0 + total_nodes.clamp_min(1.0))
        if torch.is_tensor(x_list):
            phi = torch.where(valid_seq, phi, torch.zeros_like(phi))
        return phi


def _pack_task_info_batch(task_info: Dict | List[Dict], batch_size: int, device: torch.device, scene_type_id: int) -> Dict:
    if isinstance(task_info, (list, tuple)):
        infos = list(task_info)
        if len(infos) != batch_size:
            raise ValueError("task_info list length must match batch size.")
    else:
        infos = [task_info for _ in range(batch_size)]

    packed = {"batched": True, "batch_size": batch_size}

    if scene_type_id == 1:
        max_len = 0
        for info in infos:
            max_len = max(max_len, len(info.get("x", []) or []))
        max_len = max(max_len, 2)
        x_mat = torch.zeros((batch_size, max_len), device=device, dtype=torch.float32)
        y_upper_mat = torch.zeros_like(x_mat)
        y_lower_mat = torch.zeros_like(x_mat)
        lengths = torch.zeros((batch_size,), device=device, dtype=torch.long)
        x_min = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        x_max = torch.zeros((batch_size,), device=device, dtype=torch.float32)

        for i, info in enumerate(infos):
            x_list = info.get("x", []) or []
            y_upper_list = info.get("y_upper", []) or []
            y_lower_list = info.get("y_lower", []) or []
            n = min(len(x_list), len(y_upper_list), len(y_lower_list))
            if n <= 0:
                continue
            n = max(n, 2)
            x_vals = torch.tensor(x_list[:n], device=device, dtype=torch.float32)
            y_upper_vals = torch.tensor(y_upper_list[:n], device=device, dtype=torch.float32)
            y_lower_vals = torch.tensor(y_lower_list[:n], device=device, dtype=torch.float32)
            x_mat[i, :n] = x_vals
            y_upper_mat[i, :n] = y_upper_vals
            y_lower_mat[i, :n] = y_lower_vals
            lengths[i] = n
            x_min[i] = x_vals.min()
            x_max[i] = x_vals.max()

        packed.update(
            {
                "x": x_mat,
                "y_upper": y_upper_mat,
                "y_lower": y_lower_mat,
                "lengths": lengths,
                "x_min": x_min,
                "x_max": x_max,
            }
        )

    if scene_type_id == 2:
        ranges = [float(info.get("range", 0.0)) for info in infos]
        node_counts = [
            float(info["node_count"]) if ("node_count" in info and info["node_count"] is not None) else float("nan")
            for info in infos
        ]
        packed["range"] = torch.tensor(ranges, device=device, dtype=torch.float32)
        packed["node_count"] = torch.tensor(node_counts, device=device, dtype=torch.float32)

    return packed


def _slice_task_info_batch(task_info: Dict, b0: int, b1: int) -> Dict:
    if not isinstance(task_info, dict) or not task_info.get("batched"):
        return task_info
    sliced = {"batched": True, "batch_size": b1 - b0}
    for key, val in task_info.items():
        if key in ("batched", "batch_size"):
            continue
        if torch.is_tensor(val) and val.shape[0] == task_info["batch_size"]:
            sliced[key] = val[b0:b1]
        else:
            sliced[key] = val
    return sliced

def evaluation_main_batched(
    scene_type_id: int,
    dt: float,
    net: Net,
    parameters_dicts_batch: List[Dict],
    task_info: Dict,
    t_max: float,
    gpu_id: Tuple[int, ...] = (0,),
    micro_batch_size: int | None = None,
    amp_dtype: str = "bf16",
    collect_series: bool = False,
    state_save_count: int = 0,
    state_save_dir: str | None = None,
    save_state_data: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor | Tuple[torch.Tensor, List[Dict[str, List[float]]]]:
    parameters_dicts_batch = list(parameters_dicts_batch)
    B_total = int(len(parameters_dicts_batch))
    if scene_type_id == 1:
        num_result = 5
    elif scene_type_id == 2:
        num_result = 4
    elif scene_type_id == 3:
        num_result = 2
    else:
        raise ValueError("scene_type_id error")

    if scene_type_id not in (1, 2):
        raise ValueError("Only Crossing (1) and Search-and-Rescue (2) are supported.")
    if not parameters_dicts_batch:
        raise ValueError("At least one parameter dictionary is required.")
    if dt <= 0 or t_max <= 0:
        raise ValueError("dt and t_max must be positive.")
    device = torch.device(device) if device is not None else torch.device(
        f"cuda:{int(gpu_id[0]) if gpu_id else 0}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    amp = _amp_dtype_from_str(amp_dtype)
    mb = int(micro_batch_size) if micro_batch_size is not None else B_total
    mb = max(1, min(mb, B_total))

    static = NetStatic(net, device=device)
    task_info_batched = _pack_task_info_batch(task_info, B_total, device, scene_type_id)
    out_cpu = torch.zeros((B_total, num_result), dtype=torch.float32, device="cpu")
    series_out: List[Dict[str, List[float]]] | None = None
    if collect_series:
        series_out = [{} for _ in range(B_total)]
    state_total = max(0, int(state_save_count))
    if state_total > 0 and state_save_dir:
        os.makedirs(state_save_dir, exist_ok=True)

    steps = int(math.ceil(float(t_max) / float(dt)))
    steps_f = float(max(steps, 1))
    state_init_full = getattr(net, "state_b", None)
    task_init_full = getattr(net, "task_b", None)
    if state_init_full is not None:
        if int(state_init_full.shape[0]) != B_total:
            raise ValueError("Batch initial states must match parameters_dicts_batch length.")
    if task_init_full is not None:
        if int(task_init_full.shape[0]) != B_total:
            raise ValueError("Batch initial tasks must match parameters_dicts_batch length.")

    with torch.no_grad():
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp, enabled=(amp is not None))
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            for b0 in range(0, B_total, mb):
                b1 = min(b0 + mb, B_total)
                params_mb = parameters_dicts_batch[b0:b1]
                tables = build_param_tables_batch(static, params_mb)
                state_init_mb = state_init_full[b0:b1] if state_init_full is not None else None
                task_init_mb = task_init_full[b0:b1] if task_init_full is not None else None
                nb = NetBatch(static, batch_size=(b1 - b0), state_init_b=state_init_mb, task_init_b=task_init_mb)
                dyn = ForceDynamicsBatch(static, dt=dt, scene_type_id=scene_type_id, tables=tables)
                eva = PerformanceEvaluationBatch(static)
                task_info_mb = _slice_task_info_batch(task_info_batched, b0, b1)
                state_local_count = max(0, min(nb.B, state_total - b0)) if (state_total > 0 and state_save_dir) else 0
                series_mb = None
                if collect_series:
                    if scene_type_id == 1:
                        series_mb = {
                            "phi_collision": [[] for _ in range(nb.B)],
                            "phi_correlation": [[] for _ in range(nb.B)],
                            "phi_task": [[] for _ in range(nb.B)],
                            "phi_gather": [[] for _ in range(nb.B)],
                            "connected_largest_component_size": [[] for _ in range(nb.B)],
                            "connected_component_count": [[] for _ in range(nb.B)],
                        }
                    elif scene_type_id == 2:
                        series_mb = {
                            "phi_collision": [[] for _ in range(nb.B)],
                            "phi_task": [[] for _ in range(nb.B)],
                            "phi_gather": [[] for _ in range(nb.B)],
                            "connected_min_component_size": [[] for _ in range(nb.B)],
                            "connected_component_count": [[] for _ in range(nb.B)],
                        }
                    else:
                        series_mb = {
                            "phi_collision": [[] for _ in range(nb.B)],
                        }

                def _append_series(series_lists: List[List[float]], values: torch.Tensor) -> None:
                    vals = values.detach().to("cpu", dtype=torch.float32).tolist()
                    for idx, val in enumerate(vals):
                        series_lists[idx].append(float(val))

                net_build_batched(static, nb, tables)
                pos_hist = vel_hist = task_hist = None
                if state_local_count > 0:
                    pos_hist = torch.empty(
                        (state_local_count, steps + 1, static.N, 2),
                        dtype=torch.float32,
                        device=device,
                    )
                    vel_hist = torch.empty(
                        (state_local_count, steps + 1, static.N, 2),
                        dtype=torch.float32,
                        device=device,
                    )
                    task_hist = torch.empty(
                        (state_local_count, steps + 1, static.N),
                        dtype=torch.float32,
                        device=device,
                    )
                    init_state = nb.state_b[:state_local_count]
                    pos_hist[:, 0].copy_(init_state[:, :, IDX_POS])
                    vel_hist[:, 0].copy_(init_state[:, :, IDX_VEL])
                    task_hist[:, 0].copy_(nb.task_b[:state_local_count])

                if scene_type_id == 1:
                    non_env = static.non_env_mask
                    avg_x_b = torch.nanmean(nb.state_b[:, non_env, 0], dim=1)
                    sum_f1_corr = torch.zeros((nb.B,), device=device, dtype=torch.float32)
                    sum_coll = torch.zeros((nb.B,), device=device, dtype=torch.float32)
                    sum_task = torch.zeros((nb.B,), device=device, dtype=torch.float32)
                    sum_gather = torch.zeros((nb.B,), device=device, dtype=torch.float32)

                    for si in range(steps):
                        dyn.step(nb)
                        if pos_hist is not None:
                            step_state = nb.state_b[:state_local_count]
                            pos_hist[:, si + 1].copy_(step_state[:, :, IDX_POS])
                            vel_hist[:, si + 1].copy_(step_state[:, :, IDX_VEL])
                            task_hist[:, si + 1].copy_(nb.task_b[:state_local_count])
                        t = (si + 1) * float(dt)
                        phi_coll = eva.phi_collision_avoidance(nb)
                        phi_corr = eva.phi_correlation(nb, static.comm_net_id)
                        phi_task = eva.phi_task_crossing(nb, avg_x_b, task_info_mb, t=t)
                        phi_gather = eva.phi_gather_crossing(
                            nb,
                            static.comm_net_id,
                            min_node_count=static.non_env_count,
                        )
                        comp_max, _comp_min, comp_count = eva.connectivity_component_sizes(nb, static.comm_net_id)

                        sum_coll += phi_coll
                        sum_f1_corr += _f1_torch(phi_corr, 1.0, 1.0)
                        sum_task += phi_task
                        sum_gather += phi_gather
                        if series_mb is not None:
                            _append_series(series_mb["phi_collision"], phi_coll)
                            _append_series(series_mb["phi_correlation"], phi_corr)
                            _append_series(series_mb["phi_task"], phi_task)
                            _append_series(series_mb["phi_gather"], phi_gather)
                            _append_series(series_mb["connected_largest_component_size"], comp_max)
                            _append_series(series_mb["connected_component_count"], comp_count)

                        net_build_batched(static, nb, tables)

                    mean_coll = sum_coll / steps_f
                    mean_f1_corr = sum_f1_corr / steps_f
                    mean_task = sum_task / steps_f
                    mean_gather = sum_gather / steps_f

                    fitness_corr = mean_f1_corr
                    fitness_coll = _f3_torch(mean_coll, 0.00003)
                    fitness_task = mean_task
                    fitness_gather = _f3_torch(1.0 - mean_gather, 0.01)
                    final = fitness_corr * fitness_coll * fitness_task * fitness_gather + 1e-11

                    res = torch.stack([fitness_corr, fitness_coll, fitness_task, fitness_gather, final], dim=1)
                    out_cpu[b0:b1] = res.detach().to("cpu", dtype=torch.float32)

                elif scene_type_id == 2:
                    sum_coll = torch.zeros((nb.B,), device=device, dtype=torch.float32)
                    sum_task = torch.zeros((nb.B,), device=device, dtype=torch.float32)
                    sum_gather = torch.zeros((nb.B,), device=device, dtype=torch.float32)
                    t_complete = torch.full((nb.B,), float(t_max), device=device, dtype=torch.float32)

                    for si in range(steps):
                        dyn.step(nb)
                        dyn.update_task_info(nb)
                        if pos_hist is not None:
                            step_state = nb.state_b[:state_local_count]
                            pos_hist[:, si + 1].copy_(step_state[:, :, IDX_POS])
                            vel_hist[:, si + 1].copy_(step_state[:, :, IDX_VEL])
                            task_hist[:, si + 1].copy_(nb.task_b[:state_local_count])
                        phi_coll = eva.phi_collision_avoidance(nb)
                        phi_task = eva.phi_task_searchandrescue(nb, task_info_mb)
                        phi_gather = eva.phi_gather_searchandrescue(
                            nb,
                            comm_net_id=static.comm_net_id,
                            min_node_count=5,
                        )
                        _comp_max, comp_min, comp_count = eva.connectivity_component_sizes(nb, static.comm_net_id)

                        sum_coll += phi_coll
                        sum_task += phi_task
                        sum_gather += phi_gather

                        if series_mb is not None:
                            _append_series(series_mb["phi_collision"], phi_coll)
                            _append_series(series_mb["phi_task"], phi_task)
                            _append_series(series_mb["phi_gather"], phi_gather)
                            _append_series(series_mb["connected_min_component_size"], comp_min)
                            _append_series(series_mb["connected_component_count"], comp_count)

                        t = (si + 1) * float(dt)
                        newly_done = (phi_task >= 1.0) & (t_complete >= float(t_max) - 1e-12)
                        t_complete = torch.where(newly_done, torch.full_like(t_complete, float(t)), t_complete)

                        net_build_batched(static, nb, tables)

                    mean_coll = sum_coll / steps_f
                    mean_task = sum_task / steps_f
                    mean_gather = sum_gather / steps_f
                    fitness_coll = _f3_torch(mean_coll, 0.001)
                    fitness_gather = _f3_torch(1.0 - mean_gather, 0.01)
                    fitness_task = mean_task / (1.0 + t_complete / float(max(t_max, 1e-9)))
                    final = fitness_coll * fitness_gather * fitness_task + 1e-11

                    res = torch.stack([fitness_coll, fitness_task, fitness_gather, final], dim=1)
                    out_cpu[b0:b1] = res.detach().to("cpu", dtype=torch.float32)

                else:
                    sum_coll = torch.zeros((nb.B,), device=device, dtype=torch.float32)

                    for si in range(steps):
                        dyn.step(nb)
                        dyn.update_task_info(nb)
                        if pos_hist is not None:
                            step_state = nb.state_b[:state_local_count]
                            pos_hist[:, si + 1].copy_(step_state[:, :, IDX_POS])
                            vel_hist[:, si + 1].copy_(step_state[:, :, IDX_VEL])
                            task_hist[:, si + 1].copy_(nb.task_b[:state_local_count])
                        phi_coll = eva.phi_collision_avoidance(nb)
                        sum_coll += phi_coll

                        if series_mb is not None:
                            _append_series(series_mb["phi_collision"], phi_coll)

                        net_build_batched(static, nb, tables)

                    mean_coll = sum_coll / steps_f
                    fitness_coll = _f3_torch(mean_coll, 0.001)
                    final = fitness_coll + 1e-11
                    res = torch.stack([fitness_coll, final], dim=1)
                    out_cpu[b0:b1] = res.detach().to("cpu", dtype=torch.float32)

                if state_local_count > 0 and state_save_dir:
                    for local_b in range(state_local_count):
                        global_b = b0 + local_b
                        payload = {
                            "pos": pos_hist[local_b].detach().to("cpu") if pos_hist is not None else None,
                            "vel": vel_hist[local_b].detach().to("cpu") if vel_hist is not None else None,
                            "task": task_hist[local_b].detach().to("cpu") if task_hist is not None else None,
                            "type_id": static.type_id.detach().to("cpu"),
                            "type_name_to_id": static.type_name_to_id,
                            "id_to_type_name": static.id_to_type_name,
                        }
                        if scene_type_id != 1:
                            payload["group_index"] = static.group_index.detach().to("cpu", dtype=torch.long)
                        label = f"b{global_b:03d}"
                        if save_state_data:
                            save_state_series(state_save_dir, scene_type_id, payload, label=label)

                if collect_series and series_mb is not None and series_out is not None:
                    for local_b in range(nb.B):
                        series_out[b0 + local_b] = {k: v[local_b] for k, v in series_mb.items()}

    if collect_series:
        return out_cpu, series_out or []
    return out_cpu


def evaluation_main(
    scene_type_id,
    dt,
    net,
    parameters_dicts_batch,
    task_info,
    t_max,
    cpu_num=10,
    gpu_id=(0,),
):
    if isinstance(gpu_id, (list, tuple)):
        gpu_tuple = tuple(gpu_id)
    else:
        gpu_tuple = (int(gpu_id),)
    return evaluation_main_batched(
        scene_type_id=scene_type_id,
        dt=dt,
        net=net,
        parameters_dicts_batch=parameters_dicts_batch,
        task_info=task_info,
        t_max=t_max,
        gpu_id=gpu_tuple,
    )


