"""Generate reproducible Crossing and fixed-initial Search-and-Rescue inputs."""

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile

import numpy as np

from ._scene_fixed import build_agent_node
from ._scene_parameters import generate_fixed_parameters_searchandresque
from ._scene_swarm import generate_large_swarm

ROOT = Path(__file__).resolve().parents[1]


def fixed_search_robots(rc, ra, rs):
    """Reproduce scale-3 robot states used by the original FixInitial pipeline."""
    center = (0.1, 0.1)
    parameters = generate_fixed_parameters_searchandresque()
    result = []
    for node in generate_large_swarm(center, "heter_1", 3, 8):
        vector = np.array([node["pos"][0] - center[0], node["pos"][1] - center[1]])
        velocity = tuple(vector / (np.linalg.norm(vector) + 1e-8) * 5)
        result.append(build_agent_node(
            node["type"], node["pos"], velocity, parameters,
            range_override={"rc": rc, "ra": ra, "rs": rs},
            group_index=node.get("group_index"),
        ))
    return result


def config_ranges(name):
    parts = name.split("_")
    return {key: int(next(part[len(key):] for part in parts if part.startswith(key))) for key in ("rc", "ra", "rs")}


def generate(output, task="both", config_name=None, samples=1, seed=42, force=False, examples=False):
    """Write original-format node/task JSON; never overwrite without force=True.

    Samples use seed + scene_index, matching the original initialization code.
    SR environments consume the original random-initialization sequence before
    replacing robot states with the deterministic scale-3 FixInitial formation.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 <= seed <= 2**32 - 1 - samples:
        raise ValueError("seed + samples must fit NumPy's unsigned 32-bit seed range")
    if task not in ("both", "crossing", "searchandrescue"):
        raise ValueError("Unknown task")
    output = Path(output).resolve()
    configs = sorted(path.stem for path in (ROOT / "optimized_params").glob("*.json"))
    if config_name is not None:
        if config_name not in configs:
            raise ValueError(f"Unknown configuration: {config_name}")
        if task != "both" and not config_name.startswith(task + "_"):
            raise ValueError("--task and --config select different tasks")
        configs = [config_name]
    elif task != "both":
        configs = [name for name in configs if name.startswith(task + "_")]
    if examples:
        if config_name is not None:
            raise ValueError("Use either --examples or --config")
        configs = [name for name in configs if name.endswith("_rc60_ra10_rs30")]
    filenames = [f"{name}_{i}.json" for name in configs for i in range(1, samples + 1)]
    destinations = [output / kind / filename for kind in ("NodeInitialLocationData", "TaskInfo") for filename in filenames]
    destinations.append(output / "generation.json")
    if not force:
        existing = next((path for path in destinations if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Output already exists: {existing}; use a new directory or --force")
    for path in destinations:
        if path.exists() and not path.is_file():
            raise ValueError(f"Output is not a regular file: {path}")
    with tempfile.TemporaryDirectory(prefix="mndf-scenes-") as temporary:
        staging = Path(temporary)
        for current in ("crossing", "searchandrescue"):
            names = [name for name in configs if name.startswith(current + "_")]
            if not names:
                continue
            variants = []
            for name in names:
                ranges = config_ranges(name)
                variants.append({"name": name[len(current) + 1:], "relay_range": ranges["rc"], "action_range": ranges["ra"], "sensing_range": ranges["rs"]})
            # Original helper functions print per-file progress; keep the CLI compact.
            with contextlib.redirect_stdout(io.StringIO()):
                if current == "crossing":
                    from ._scene_crossing import generate_and_save_nodes_crossing
                    generate_and_save_nodes_crossing("heter_1", 3, str(staging / "NodeInitialLocationData"), str(staging / "TaskInfo"), samples, variants, seed)
                else:
                    from ._scene_searchandrescue import generate_and_save_nodes_searchandrescue
                    generate_and_save_nodes_searchandrescue((0.1, 0.1), "heter_1", 3, str(staging / "NodeInitialLocationData"), str(staging / "TaskInfo"), 60, samples, variants, seed)
            if current == "searchandrescue":
                for name in names:
                    ranges = config_ranges(name)
                    fixed = fixed_search_robots(**ranges)
                    for i in range(1, samples + 1):
                        path = staging / "NodeInitialLocationData" / f"{name}_{i}.json"
                        nodes = json.loads(path.read_text())
                        nodes = copy.deepcopy(fixed) + [node for node in nodes if "environment" in node["type"]]
                        for index, node in enumerate(nodes):
                            node["id"] = index
                        path.write_text(json.dumps(nodes, indent=4) + "\n")
        # Generation must complete before existing output files can be replaced.
        for kind in ("NodeInitialLocationData", "TaskInfo"):
            (output / kind).mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                (output / kind / filename).write_bytes((staging / kind / filename).read_bytes())
    manifest = {"base_seed": seed, "scene_seed_rule": "base_seed + scene_index", "samples_per_configuration": samples,
                "configurations": configs, "swarm_count": 3, "robots_per_swarm": 21,
                "searchandrescue_initialization": "fixed scale-3 robot states; original random environments",
                "json_pairs": len(filenames)}
    (output / "generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("both", "crossing", "searchandrescue"), default="both")
    parser.add_argument("--config", help="Generate one configuration instead of the full selected task grid")
    parser.add_argument("--examples", action="store_true", help="Generate only the two configurations shipped in examples/")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace matching output files")
    args = parser.parse_args(argv)
    try:
        result = generate(args.output, args.task, args.config, args.samples, args.seed, args.force, args.examples)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
