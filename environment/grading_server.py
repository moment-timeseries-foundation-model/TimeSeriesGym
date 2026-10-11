import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from timeseriesgym.grade import validate_submission
from timeseriesgym.registry import registry

app = Flask(__name__)

PRIVATE_DATA_DIR = "/private/data"
COMPETITION_ID = os.getenv("COMPETITION_ID")  # This is populated for us at container runtime


def run_validation(submission: Path) -> str:
    new_registry = registry.set_data_dir(Path(PRIVATE_DATA_DIR))
    competition = new_registry.get_competition(COMPETITION_ID)
    is_valid, message = validate_submission(submission, competition)
    return message


ALLOWED_SUFFIXES = {".csv", ".npy", ".h5", ".hdf5", ".pkl", ".py"}


@app.route("/validate", methods=["POST"])
def validate():
    submission_file = request.files["file"]
    suffix = Path(submission_file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return jsonify({"result": f"Unsupported submission file type {suffix!r}."}), 400
    # A new root-only directory per upload: the agent can write to /tmp, so it could plant or
    # swap a fixed path there that this root process then writes and reads, and concurrent
    # uploads would overwrite each other. Keep the suffix: validation checks it against the
    # sample submission's.
    with tempfile.TemporaryDirectory(prefix="validate-") as upload_dir:
        submission_path = Path(upload_dir) / f"submission{suffix}"
        submission_file.save(submission_path)

        try:
            result = run_validation(submission_path)
        except Exception as e:
            # Server error
            return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

    return jsonify({"result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
