import subprocess


def get_gpu_generation() -> str | None:
    """Returns the GPU generation, if available."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    generation = result.stdout.strip().split("\n")

    if not generation:
        return None

    return ", ".join([info.strip() for info in generation])


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")
