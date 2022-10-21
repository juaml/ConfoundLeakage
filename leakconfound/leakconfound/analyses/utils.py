from pathlib import Path


def save_paper_val(base_dir, experiment, confound, name, value, precision=2):
    path = Path(f"{base_dir}")
    path.mkdir(parents=True, exist_ok=True)
    path = f"{base_dir}{experiment}_{confound}_{name}"
    with open(path, 'w') as f:
        f.write(str(round(value, precision)))
