"""Small, portable entry point for simulation and CMA-ES optimization."""

import argparse
import json
import sys
from pathlib import Path

import cma
import numpy as np
import torch

from .dynamics import Net, evaluation_main_batched

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    1: ("correlation_fitness", "collision_fitness", "task_fitness", "gather_fitness", "objective_value"),
    2: ("collision_fitness", "task_fitness", "gather_fitness", "objective_value"),
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve_settings(config_name):
    task = config_name.split("_", 1)[0]
    if task not in ("crossing", "searchandrescue"):
        raise ValueError("Config name must start with crossing_ or searchandrescue_.")
    if not (ROOT / "optimized_params" / f"{config_name}.json").is_file():
        raise ValueError(f"Unknown configuration: {config_name}")
    return load_json(ROOT / "configs" / f"{task}.json")


def load_scene(data_dir, config_name, scene_index, scene_type):
    filename = f"{config_name}_{scene_index}.json"
    net = Net(device=torch.device("cpu"))
    net.load_nodes_from_json(str(Path(data_dir) / "NodeInitialLocationData" / filename), scene_type)
    return net, load_json(Path(data_dir) / "TaskInfo" / filename)


def evaluate(net, task, parameters, settings, device, batch_size):
    result = evaluation_main_batched(
        scene_type_id=settings["scene_type_id"], dt=settings["dt"], net=net,
        parameters_dicts_batch=parameters, task_info=task, t_max=settings["t_max"],
        micro_batch_size=batch_size, amp_dtype="bf16", device=device,
    )
    if not torch.isfinite(result).all():
        raise RuntimeError("Simulation produced non-finite fitness values.")
    return result


def parameter_entries(bounds):
    for section, groups in bounds.items():
        for group, values in groups.items():
            for name, interval in values.items():
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(f"Expected [lower, upper] for {section}.{group}.{name}")
                if not np.isfinite(interval).all() or interval[0] >= interval[1]:
                    raise ValueError(f"Invalid parameter interval: {interval}")
                yield section, group, name, interval


def decode_parameters(entries, solutions):
    lower = torch.tensor([entry[3][0] for entry in entries], dtype=torch.float32)
    upper = torch.tensor([entry[3][1] for entry in entries], dtype=torch.float32)
    values = torch.tensor(np.asarray(solutions), dtype=torch.float32) * (upper - lower) + lower
    result = []
    for row in values:
        parameter = {}
        for (section, group, name, _), value in zip(entries, row):
            parameter.setdefault(section, {}).setdefault(group, {})[name] = float(value)
        result.append(parameter)
    return result


def optimize(args, settings):
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Optimization output must be outside the release directory.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory to preserve previous optimization results.")
    count = settings["initial_net_count"]
    scenes = [load_scene(args.data_dir, args.config, i, settings["scene_type_id"]) for i in range(1, count + 1)]
    entries = list(parameter_entries(load_json(ROOT / "Parameter" / f"{args.config}.json")))
    seed = settings["random_seed"]
    np.random.seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "settings.json", {**settings, "config": args.config, "data_dir": str(args.data_dir.resolve()), "device": args.device})
    strategy = cma.CMAEvolutionStrategy(np.random.uniform(0, 1, len(entries)), settings["sigma"], {
        "popsize": settings["population_size"], "bounds": [0.0, 1.0],
        "AdaptSigma": True, "CMA_eigenmethod": np.linalg.eigh,
        "CMA_active_injected": True, "CMA_elitist": True, "seed": seed,
        "verb_filenameprefix": str(output / "cmaes_"),
    })
    candidates, history = [], []
    net, task = scenes[0]
    for generation in range(settings["max_generations"]):
        solutions = strategy.ask()
        parameters = decode_parameters(entries, solutions)
        metrics = evaluate(net, task, parameters, settings, args.device, settings["batch_size"])
        losses = -metrics[:, -1].numpy()
        strategy.tell(solutions, losses.tolist())
        candidates.extend((solution.copy(), float(loss)) for solution, loss in zip(solutions, losses))
        history.append({"generation": generation, "best_objective": float(-losses.min()), "population_size": len(solutions)})
        save_json(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
    selected, seen = [], set()
    for solution, _ in sorted(candidates, key=lambda item: item[1]):
        key = tuple(float(v) for v in solution)
        if key not in seen:
            seen.add(key)
            selected.append(solution)
        if len(selected) >= settings["selection_size"]:
            break
    parameters = decode_parameters(entries, selected)
    scores = np.zeros((len(selected), count), dtype=np.float32)
    for i, (net, task) in enumerate(scenes):
        scores[:, i] = evaluate(net, task, parameters, settings, args.device, settings["batch_size"])[:, -1].numpy()
        print(f"candidate selection scene {i + 1}/{count}", flush=True)
    best = int(np.argmax(scores.mean(axis=1)))
    save_json(output / f"{args.config}.json", parameters[best])
    save_json(output / "selection.json", {"candidate_count": len(selected), "best_mean_objective": float(scores[best].mean()), "best_scene_objectives": scores[best].tolist()})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        from .scenes import main as generate_main
        generate_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__, epilog="Scene generation: python -m mndf generate --help")
    parser.add_argument("command", choices=("simulate", "optimize", "generate"))
    parser.add_argument("--config", required=True, help="One of the 54 configuration stems in optimized_params/")
    parser.add_argument("--data-dir", type=Path, help="Directory containing NodeInitialLocationData/ and TaskInfo/")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--params", type=Path, help="Optional custom optimized parameter file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--t-max", type=float)
    parser.add_argument("--generations", type=int)
    args = parser.parse_args()
    settings = resolve_settings(args.config)
    if args.t_max is not None:
        settings["t_max"] = args.t_max
    if args.generations is not None:
        settings["max_generations"] = args.generations
    if settings["t_max"] <= 0 or settings["max_generations"] <= 0 or args.scene <= 0:
        parser.error("Time limit, generation count, and scene index must be positive.")
    if args.command == "optimize":
        if args.data_dir is None or args.output is None:
            parser.error("Optimization requires --data-dir and --output.")
        if args.params is not None:
            parser.error("Optimization starts from seeded random parameters; --params is for simulation.")
        optimize(args, settings)
    else:
        data_dir = args.data_dir or ROOT / "examples"
        net, task = load_scene(data_dir, args.config, args.scene, settings["scene_type_id"])
        parameter = load_json(args.params or ROOT / "optimized_params" / f"{args.config}.json")
        values = evaluate(net, task, [parameter], settings, args.device, settings["batch_size"])[0].tolist()
        result = dict(zip(METRICS[settings["scene_type_id"]], values))
        print(json.dumps(result, indent=2))
        if args.output:
            destination = args.output.resolve()
            if destination == ROOT or ROOT in destination.parents:
                parser.error("Save simulation output outside the release directory.")
            if destination.exists():
                parser.error("Output file already exists.")
            save_json(destination, result)


if __name__ == "__main__":
    main()
