"""Descriptions state the facts of the data their prepare scripts make.

Each test reads the fact from the prepare script (with `ast`, without running it) and checks that
the description states it, so the two cannot drift apart silently.
"""

import ast
import importlib
from pathlib import Path

import pytest


def _directory(competition_id: str) -> Path:
    module = importlib.import_module(f"timeseriesgym.competitions.{competition_id}.prepare")
    return Path(module.__file__).parent


def _assigned(tree: ast.AST, name: str):
    """The literal value assigned to `name` in the prepare script (exactly one assignment)."""
    values = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    assert len(values) == 1, name
    return values[0]


def _mit_bih_prepare() -> ast.AST:
    return ast.parse((_directory("MIT-BIH-Arrhythmia") / "prepare.py").read_text())


def _normal_beat_keep_probability(tree: ast.AST) -> float:
    """The p of `np.random.random() < p`, with which a normal beat is kept."""
    found = [
        node.comparators[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and ast.unparse(node.left) == "np.random.random()"
        and isinstance(node.ops[0], ast.Lt)
        and isinstance(node.comparators[0], ast.Constant)
    ]
    assert len(found) == 1
    return found[0]


def test_mit_bih_description_names_the_held_out_test_recordings():
    tree = _mit_bih_prepare()
    records, test = _assigned(tree, "record_numbers"), _assigned(tree, "test_records")
    text = " ".join((_directory("MIT-BIH-Arrhythmia") / "description.md").read_text().split())
    listed = ", ".join(test[:-1]) + f" and {test[-1]}"
    # the description did not say that the split is by recording
    assert f"They come from {len(test)} of the {len(records)} recordings (records {listed})" in text
    assert f"the training beats from the other {len(records) - len(test)}" in text
    assert "no recording is in both" in text


@pytest.mark.parametrize("name", ["description_only_label.md", "description_label_and_model.md"])
def test_weak_supervision_descriptions_state_how_the_beats_were_made(name):
    tree = _mit_bih_prepare()
    records = len(_assigned(tree, "record_numbers"))
    train = records - len(_assigned(tree, "test_records"))
    directory = _directory("MIT-BIH-Arrhythmia-Weak-Supervision")
    text = " ".join((directory / name).read_text().split())
    # the descriptions said neither how long the windows are nor how the beats were sampled
    assert f"a window of {_assigned(tree, 'segment_len')} consecutive samples" in text
    p = _normal_beat_keep_probability(tree)
    assert f"each normal-beat window only with probability {p:g}" in text
    assert f"the beats come from {train} of the database's {records} recordings" in text
    assert "standardized" in text and "no RR intervals, beat times or record (patient) ids" in text
    # heuristics that need beat times or record ids are marked as not computable here
    assert "1. R-wave appears earlier than usual (this one cannot be computed here" in text
    assert "compute subject-specific thresholds for each heuristic automatically" not in text
    assert 'to identify an "early R-wave"' not in text
    assert "check on the data whether, and in which direction, a heuristic separates" in text
