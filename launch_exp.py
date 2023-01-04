import os
import re
from pathlib import Path
import sys
from minydra import resolved_args
from yaml import safe_load
import subprocess
from sbatch import now
import copy


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary
    as a value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share
        the same key.

    Returns
    -------
    return_dict_and_duplicates: tuple(dict, list(str))
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            elif isinstance(v, list) and isinstance(dict1[k], list):
                if len(dict1[k]) != len(dict2[k]):
                    raise ValueError(
                        f"List for key {k} has different length in dict1 and dict2."
                        + " Use an empty dict {} to pad for items in the shorter list."
                    )
                return_dict[k] = [merge_dicts(d1, d2)[0] for d1, d2 in zip(dict1[k], v)]
            else:
                return_dict[k] = dict2[k]

    return return_dict


def get_commit():
    try:
        commit = (
            subprocess.check_output("git rev-parse --verify HEAD".split())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "unknown"
    return commit


def find_exp(name):
    exp_dir = Path(__file__).parent / "configs" / "exps"
    exp_file = exp_dir / f"{name}.yml"
    if exp_file.exists():
        return exp_file
    exp_file = exp_dir / f"{name}.yaml"
    if exp_file.exists():
        return exp_file

    raise ValueError(f"Could not find experiment {name}")


def seconds_to_time_str(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def cli_arg(args, key=""):
    s = ""
    for k, v in args.items():
        parent = "" if not key else f"{key}."
        if isinstance(v, dict):
            s += cli_arg(v, key=f"{parent}{k}")
        else:
            if " " in str(v) or "," in str(v) or isinstance(v, str):
                if "'" in str(v) and '"' in str(v):
                    v = str(v).replace("'", "\\'")
                    v = f"'{v}'"
                elif "'" in str(v):
                    v = f'"{v}"'
                else:
                    v = f"'{v}'"
            s += f" --{parent}{k}={v}"
    return s


if __name__ == "__main__":
    args = resolved_args()
    assert "exp" in args
    regex = args.get("match", ".*")

    exp_name = args.exp.replace(".yml", "").replace(".yaml", "")
    exp_file = find_exp(exp_name)

    exp = safe_load(exp_file.open("r"))

    runs = exp["runs"]

    commands = []

    for run in runs:
        params = exp["default"].copy()
        job = merge_dicts(exp["job"].copy(), run.pop("job", {}))
        if run.pop("_no_exp_default_", False):
            params = {}
        params = merge_dicts(params, run)
        if "time" in job:
            job["time"] = seconds_to_time_str(job["time"])

        if "wandb_tags" in params:
            params["wandb_tags"] += "," + exp_name
        else:
            params["wandb_tags"] = exp_name

        py_args = f'py_args="{cli_arg(params).strip()}"'

        sbatch_args = " ".join(
            [f"{k}={v}" for k, v in job.items()] + [f"exp_name={exp_name}"]
        )
        command = f"python sbatch.py {sbatch_args} {py_args}"
        commands.append(command)

    commands = [c for c in commands if re.findall(regex, c)]

    print(f"🔥 About to run {len(commands)} jobs:\n\n • " + "\n\n  • ".join(commands))

    separator = "\n" * 4 + f"{'#' * 80}\n" * 4 + "\n" * 4
    text = "<><><> Experiment command: $ " + " ".join(["python"] + sys.argv)
    text += "\n<><><> Experiment commit: " + get_commit()
    text += "\n<><><> Experiment config:\n\n-----" + exp_file.read_text() + "-----"
    text += "\n<><><> Experiment runs:\n\n • " + "\n\n  • ".join(commands) + separator

    confirm = input("\n🚦 Confirm? [y/n]")

    if confirm == "y":
        outputs = [
            print(f"Launching job {c:3}", end="\r") or os.popen(command).read().strip()
            for c, command in enumerate(commands)
        ]
        outdir = Path(__file__).resolve().parent / "data" / "exp_outputs" / exp_name
        outfile = outdir / f"{exp_name.split('/')[-1]}_{now()}.txt"
        outfile.parent.mkdir(exist_ok=True, parents=True)
        text += separator.join(outputs)
        jobs = [
            line.replace(sep, "").strip()
            for line in text.splitlines()
            if (sep := "Submitted batch job ") in line
        ]
        text += f"{separator}All jobs launched: {' '.join(jobs)}"
        with outfile.open("w") as f:
            f.write(text)
        print(f"Output written to {str(outfile)}")
        print("All job launched:", " ".join(jobs))
    else:
        print("Aborting")
