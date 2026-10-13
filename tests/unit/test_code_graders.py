"""Regression tests for the code-challenge graders: resnet-tensorflow-to-pytorch counts the samples
the test model is trained on, and stomp-R-to-python also checks a variant of each test case.

The graders run through the registry and `timeseriesgym.grade.grade_code`, as the `grade` CLI
runs them, on data made by the challenges' own prepare (downloads replaced) or, for stomp, on a
random series with answers from a brute-force matrix profile in R's conventions.
"""

import copy
import importlib
import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from timeseriesgym.grade import grade_code
from timeseriesgym.registry import Registry

RESNET = "resnet-tensorflow-to-pytorch"
STOMP = "stomp-R-to-python"


def _module(competition_id: str, name: str):
    return importlib.import_module(f"timeseriesgym.competitions.{competition_id}.{name}")


def _challenge_file(competition_id: str, name: str) -> Path:
    return Path(_module(competition_id, "grade").__file__).with_name(name)


# ---------------------------------------------------------------- resnet: the sample budget


@pytest.fixture(scope="module")
def resnet_data(tmp_path_factory):
    """A data dir with the resnet challenge prepared by its own prepare (no downloads)."""
    data_dir = tmp_path_factory.mktemp("data")
    competition = Registry(data_dir).get_competition(RESNET)
    module = _module(RESNET, "prepare")
    public, private = competition.public_dir, competition.private_dir
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    original = module.download_file
    module.download_file = lambda url, destination: destination.write_text("# code\n")
    try:
        module.prepare(data_dir / RESNET / "raw", public, private)
    finally:
        module.download_file = original
    return data_dir


def _grade_resnet(data_dir, tmp_path, source: str, n_epochs: int | None = None):
    """grade_code on `source`; `n_epochs` replaces the grading argument (a cheaper budget)."""
    submission = tmp_path / "submission.py"
    submission.write_text(source)
    competition = Registry(data_dir).get_competition(RESNET)
    if n_epochs is not None:
        competition.coding_config["required_functions"]["train"]["required_args"]["n_epochs"] = (
            n_epochs
        )
    return grade_code(submission, competition)


# A cheap model that is enough for this data (logistic regression on the 50 time steps), but is
# not the architecture of resnet.py.
LINEAR = """
import numpy as np
import torch
from torch import nn


class ResNet(nn.Module):
    def __init__(self, output_directory, input_shape, n_classes, n_feature_maps, verbose=False,
                 random_seed=42):
        super().__init__()
        torch.manual_seed(random_seed)
        self.fc = nn.Linear(input_shape[0] * input_shape[1], n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def train(model, train_data, val_data, batch_size, n_epochs, learning_rate, verbose):
    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(n_epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
    return model


def predict(model, test_data):
    model.eval()
    x = torch.stack([test_data[i][0] for i in range(len(test_data))])
    with torch.no_grad():
        return np.zeros(len(x), dtype=np.int64), torch.softmax(model(x), 1).numpy()
"""

# The reference port (rl_envs' oracle): BatchNorm statistics re-estimated after training, so
# that it passes after a single epoch; the budget tests grade with n_epochs=1 to stay cheap.
REFERENCE = """

import os

import numpy as np
import torch
from torch import nn
from torch.nn import Module


class _Block(nn.Module):
    '''Three Conv1D-BN(-ReLU) layers (kernels 8, 5, 3) plus a shortcut, as in Classifier_RESNET.'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding="same"),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels:  # "expand channels for the sum"
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels)
            )
        else:  # "no need to expand channels because they are equal"
            self.shortcut = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.body(x))


class ResNet(Module):
    def __init__(
        self,
        output_directory,
        input_shape,
        n_classes,
        n_feature_maps,
        verbose=False,
        random_seed=42,
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.output_directory = output_directory
        self.verbose = verbose
        n_channels = input_shape[0]
        self.blocks = nn.Sequential(
            _Block(n_channels, n_feature_maps),
            _Block(n_feature_maps, n_feature_maps * 2),
            _Block(n_feature_maps * 2, n_feature_maps * 2),
        )
        self.fc = nn.Linear(n_feature_maps * 2, n_classes)
        os.makedirs(output_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_directory, "model_init.pt"))
        if verbose:
            print(self)

    def forward(self, x):
        h = self.blocks(x).mean(dim=-1)  # GlobalAveragePooling1D
        return torch.softmax(self.fc(h), dim=-1)  # Dense(n_classes, activation="softmax")


def train(model, train_data, val_data, batch_size, n_epochs, learning_rate, verbose):
    loader = torch.utils.data.DataLoader(
        train_data, batch_size=int(min(len(train_data) / 10, batch_size)), shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=50, min_lr=1e-4
    )
    best = float("inf")
    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            probs = model(x)
            loss = nn.functional.nll_loss(
                torch.log(probs.clamp_min(1e-12)), y
            )  # categorical cross-entropy
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        epoch_loss = total / len(train_data)
        scheduler.step(epoch_loss)
        if epoch_loss < best:  # ModelCheckpoint(monitor="loss", save_best_only=True)
            best = epoch_loss
            torch.save(model.state_dict(), os.path.join(model.output_directory, "best_model.pt"))
        if verbose:
            print(f"epoch {epoch + 1}/{n_epochs} loss {epoch_loss:.4f}")
    model.load_state_dict(torch.load(os.path.join(model.output_directory, "best_model.pt")))
    _recompute_batchnorm_statistics(model, train_data, batch_size)
    return model


def _recompute_batchnorm_statistics(model, data, batch_size):
    # With few optimizer steps (n_epochs=1) the BatchNorm running averages are far from the
    # population statistics, and eval-mode predictions collapse. Re-estimate them with a
    # cumulative average over the training data (no parameter update).
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    saved = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    model.train()
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(data, batch_size=batch_size):
            model(x)
    for bn, m in zip(bns, saved, strict=False):
        bn.momentum = m


def predict(model, test_data):
    model.eval()
    xs, ys = zip(*[test_data[i] for i in range(len(test_data))], strict=False)
    x = torch.stack(xs)
    with torch.no_grad():
        y_pred = model(x).numpy()
    return np.asarray([int(y) for y in ys]), y_pred
"""

# More epochs than the grading arguments ask for (Sonnet's submission in a Claude Code
# evaluation used max(n_epochs, 30) when n_epochs was 1); one epoch more is enough to fail.
OVERTRAINED = REFERENCE.replace(
    "for epoch in range(n_epochs):", "for epoch in range(n_epochs + 1):"
)

# The tensors copied out of the dataset first (through numpy), then two passes per epoch.
COPIED_OUT = REFERENCE.replace(
    "    loader = torch.utils.data.DataLoader(\n"
    "        train_data, batch_size=int(min(len(train_data) / 10, batch_size)), shuffle=True\n"
    "    )\n",
    "    x = np.stack([train_data[i][0].numpy() for i in range(len(train_data))])\n"
    "    y = np.asarray([int(train_data[i][1]) for i in range(len(train_data))])\n"
    "    copy = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))\n"
    "    loader = torch.utils.data.DataLoader(copy, batch_size=64, shuffle=True)\n",
).replace("        for x, y in loader:\n", "        for x, y in [*loader, *loader]:\n")

# Training before `train` is called: at import, on the public training file.
PRETRAINED_AT_IMPORT = (
    REFERENCE
    + """
import tempfile

_data = torch.load(@PATH@, weights_only=False)
_net = ResNet(tempfile.mkdtemp() + "/", [1, 50], 2, 64)
train(_net, _data, None, 64, 1, 0.001, False)
_STATE = _net.state_dict()
_init = ResNet.__init__


def _pretrained_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.load_state_dict(_STATE)


ResNet.__init__ = _pretrained_init
"""
)

# Passes that train nothing, besides the reference's own BatchNorm re-estimation: validation
# each epoch, and a forward pass with gradients but no back-propagation.
EXTRA_PASSES = REFERENCE.replace(
    "        epoch_loss = total / len(train_data)\n",
    "        epoch_loss = total / len(train_data)\n"
    "        predict(model, val_data)\n"
    "        model.train()\n"
    "        for x, _ in loader:\n"
    "            model(x).argmax(1)\n",
)


# A port of the Keras model with Keras' settings: BatchNormalization(momentum=0.99, epsilon=1e-3)
# is BatchNorm1d(momentum=0.01, eps=1e-3); mini-batch min(n / 10, batch_size); Adam; the
# best-loss checkpoint. Measured in tsgym-core (CPU, 4 threads): test accuracy 0.49-0.51 on 10
# of 10 seeds after 1 epoch, 1.00 on 20 of 20 after 40 (its BatchNorm running statistics lag
# the weights for the first 20 to 30 epochs, and eval-mode predictions collapse to one class).
FAITHFUL = """

import copy
import os

import numpy as np
import torch
from torch import nn
from torch.nn import Module

BN_MOMENTUM = 0.01
BN_EPS = 1e-3


def _bn(c):
    return nn.BatchNorm1d(c, momentum=BN_MOMENTUM, eps=BN_EPS)


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cx, self.bx = nn.Conv1d(cin, cout, 8, padding="same"), _bn(cout)
        self.cy, self.by = nn.Conv1d(cout, cout, 5, padding="same"), _bn(cout)
        self.cz, self.bz = nn.Conv1d(cout, cout, 3, padding="same"), _bn(cout)
        if cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, padding="same"), _bn(cout))
        else:
            self.sc = _bn(cout)

    def forward(self, x):
        h = torch.relu(self.bx(self.cx(x)))
        h = torch.relu(self.by(self.cy(h)))
        h = self.bz(self.cz(h))
        return torch.relu(self.sc(x) + h)


class ResNet(Module):
    def __init__(
        self, output_directory, input_shape, n_classes, n_feature_maps, verbose=False,
        random_seed=42,
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.output_directory = output_directory
        f = n_feature_maps
        self.b1 = _Block(input_shape[0], f)
        self.b2 = _Block(f, 2 * f)
        self.b3 = _Block(2 * f, 2 * f)
        self.dense = nn.Linear(2 * f, n_classes)

    def forward(self, x):
        h = self.b3(self.b2(self.b1(x))).mean(dim=-1)
        return torch.softmax(self.dense(h), dim=-1)


def train(model, train_data, val_data, batch_size, n_epochs, learning_rate, verbose):
    mini = int(min(len(train_data) / 10, batch_size))
    loader = torch.utils.data.DataLoader(train_data, batch_size=mini, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50, min_lr=1e-4)
    best, best_state = float("inf"), None
    for _ in range(n_epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            p = model(x)
            loss = nn.functional.nll_loss(torch.log(p.clamp_min(1e-7)), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        epoch_loss = total / n
        sched.step(epoch_loss)
        if epoch_loss < best:
            best, best_state = epoch_loss, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(model.output_directory, exist_ok=True)
    return model


def predict(model, test_data):
    model.eval()
    x = torch.stack([test_data[i][0] for i in range(len(test_data))])
    y = np.asarray([int(test_data[i][1]) for i in range(len(test_data))])
    with torch.no_grad():
        return y, model(x).numpy()
"""

UNTRAINED = FAITHFUL.replace(
    "    mini = int(min(len(train_data) / 10, batch_size))",
    "    return model\n    mini = int(min(len(train_data) / 10, batch_size))",
)


# A ResNet without its shortcuts: the layers are defined but not used.
NO_SHORTCUT = FAITHFUL.replace(
    "        return torch.relu(self.sc(x) + h)", "        return torch.relu(h)"
)


def test_resnet_a_keras_faithful_port_passes_with_the_grading_arguments(resnet_data, tmp_path):
    # With n_epochs=1 it failed (accuracy 0.49); the grading arguments now ask for 40 epochs.
    report = _grade_resnet(resnet_data, tmp_path, FAITHFUL)
    assert report.valid_submission and report.test_metric == "1/1"


def test_resnet_an_untrained_model_fails(resnet_data, tmp_path):
    report = _grade_resnet(resnet_data, tmp_path, UNTRAINED)
    assert report.valid_submission and report.test_metric == "0/1"
    assert report.executed_functions == "2/2"


def test_resnet_a_submission_within_the_budget_passes(resnet_data, tmp_path):
    report = _grade_resnet(resnet_data, tmp_path, REFERENCE, n_epochs=1)
    assert report.valid_submission and report.test_metric == "1/1"
    assert report.executed_functions == "2/2" and report.executed_class_methods == "1/1"


@pytest.mark.parametrize("source", [OVERTRAINED, COPIED_OUT], ids=["more_epochs", "copied_tensors"])
def test_resnet_training_beyond_n_epochs_fails(resnet_data, tmp_path, source):
    # Sonnet's submission set n_epochs = max(n_epochs, 30) and passed; copying the tensors out
    # of the dataset first must not hide the extra passes.
    report = _grade_resnet(resnet_data, tmp_path, source, n_epochs=1)
    assert report.valid_submission and report.test_metric == "0/1"
    assert report.executed_functions == "2/2"  # the structural checks are unchanged


def test_resnet_training_at_import_counts(resnet_data, tmp_path):
    path = str(Registry(resnet_data).get_competition(RESNET).public_dir / "train_data.pt")
    source = PRETRAINED_AT_IMPORT.replace("@PATH@", repr(path))
    report = _grade_resnet(resnet_data, tmp_path, source, n_epochs=1)
    assert report.valid_submission and report.test_metric == "0/1"


def test_resnet_passes_without_backpropagation_do_not_count(resnet_data, tmp_path):
    report = _grade_resnet(resnet_data, tmp_path, EXTRA_PASSES, n_epochs=1)
    assert report.valid_submission and report.test_metric == "1/1"


def test_resnet_another_architecture_fails_even_when_accurate(resnet_data, tmp_path):
    # The linear model reaches accuracy 1.0 within the budget (it passed before the check).
    report = _grade_resnet(resnet_data, tmp_path, LINEAR)
    assert report.valid_submission and report.test_metric == "0/1"
    assert report.executed_functions == "2/2"


def _port(source: str):
    namespace: dict = {}
    exec(compile(source, "port", "exec"), namespace)
    return namespace["ResNet"]


@pytest.mark.parametrize(
    "source, used",
    [(FAITHFUL, 504_258), (REFERENCE, 504_258), (NO_SHORTCUT, 495_170), (LINEAR, 102)],
    ids=["faithful", "reference", "no_shortcut", "linear"],
)
def test_resnet_architecture_counts_the_parameters_the_forward_pass_uses(tmp_path, source, used):
    architecture = _module(RESNET, "architecture")
    assert architecture.keras_trainable_parameters(1, 64, 2) == 504_258  # Keras' model.summary()
    model = _port(source)(str(tmp_path) + "/", [1, 50], 2, 64)
    assert architecture.trainable_parameters_in_use(model, torch.randn(10, 1, 50)) == used
    registered = sum(p.numel() for p in model.parameters())
    assert registered == (504_258 if source is NO_SHORTCUT else used)  # defined, not used


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


def test_sample_counter_counts_backpropagated_samples_of_any_instance():
    counter_cls = _module(RESNET, "training_budget").TrainingSampleCounter
    x = torch.randn(5, 3)
    net = _Net()
    with counter_cls(lambda: _Net) as counter:
        net(x).sum().backward()  # 5
        with torch.no_grad():
            net(x)  # no gradient
        net(x)  # never back-propagated
        copy.deepcopy(net)(x).sum().backward()  # 10: a copy
        buffer = io.BytesIO()
        torch.save(net, buffer)
        buffer.seek(0)
        torch.load(buffer, weights_only=False)(x[:2]).sum().backward()  # 12: a reloaded model
        torch.nn.Linear(3, 2)(x).sum().backward()  # another class: not counted
    assert counter.samples == 12
    net(x).sum().backward()  # after the counter: not counted
    assert counter.samples == 12

    with counter_cls(lambda: None) as counter:  # the class does not exist (yet)
        net(x).sum().backward()
    assert counter.samples == 0


def test_resnet_description_states_the_grading_arguments():
    config = yaml.safe_load(_challenge_file(RESNET, "config.yaml").read_text())
    functions = config["coding_config"]["required_functions"]
    n_epochs = functions["train"]["required_args"]["n_epochs"]
    threshold = functions["predict"]["expected_accuray"]
    description = _challenge_file(RESNET, "description.md").read_text()
    assert f"n_epochs={n_epochs}," in description
    assert f"greater than {threshold:.0%}" in description
    assert "n_epochs * len(train_data)" in description
    assert "504,258 trainable parameters" in description  # keras_trainable_parameters(1, 64, 2)


# ---------------------------------------------------------------- stomp: the hidden variant

WINDOW, LENGTH = 30, 550
EPS = np.finfo(float).eps ** 0.5  # vars()$eps in tsmp


def _windows(x):
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=float), WINDOW)
    mean, sd = windows.mean(axis=1), windows.std(axis=1)
    return windows, mean, sd


def _distances(query, data):
    """dist[i, j]: z-normalized distance between query window i and data window j, as STOMP."""
    qw, qm, qs = _windows(query)
    dw, dm, ds = _windows(data)
    qt = qw @ dw.T
    d = 2 * (WINDOW - (qt - WINDOW * np.outer(qm, dm)) / np.outer(qs, ds))
    return np.sqrt(np.maximum(d, 0))


def _first_min(dist, mask):
    """Per column j: the minimum over the rows where mask[:, j], and its first 1-based row
    (Inf and -Inf where there is none), as R's strict `<` updates in row order give."""
    d = np.where(mask, dist, np.inf)
    rows = np.argmin(d, axis=0)
    best = d[rows, np.arange(d.shape[1])]
    index = np.where(np.isfinite(best), rows + 1.0, -np.inf)
    return best, index


def brute_force_stomp(ref_data, query_data, window_size, exclusion_zone=0.5, verbose=2):
    """The matrix profile in tsmp's conventions, by definition (finite inputs only)."""
    assert window_size == WINDOW
    if query_data is not None:
        dist = _distances(query_data, ref_data)
        mp, pi = _first_min(dist, np.ones_like(dist, dtype=bool))
        return mp, pi, None, None, None, None
    dist = _distances(ref_data, ref_data)
    n = dist.shape[0]
    ez = int(round(window_size * exclusion_zone + EPS))
    i, j = np.indices((n, n))
    allowed = np.abs(i - j) > ez
    mp, pi = _first_min(dist, allowed)
    rmp, rpi = _first_min(dist, allowed & (i >= j))  # neighbours at or after j
    lmp, lpi = _first_min(dist, allowed & (i <= j))  # neighbours at or before j
    return mp, pi, rmp, rpi, lmp, lpi


def _r_csv(self_join, join) -> str:
    columns = ["mp_self_mp", "mp_self_pi", "mp_self_rmp", "mp_self_rpi", "mp_self_lmp"]
    columns += ["mp_self_lpi", "mp_join_mp", "mp_join_pi"]
    frame = pd.DataFrame(np.column_stack([*self_join, *join[:2]]), columns=columns)
    return frame.to_csv(index=False).replace("-inf", "-Inf").replace("inf", "Inf")


@pytest.fixture(scope="module")
def stomp_data(tmp_path_factory):
    """A data dir with random-walk inputs and their brute-force outputs as the answers."""
    data_dir = tmp_path_factory.mktemp("data")
    competition = Registry(data_dir).get_competition(STOMP)
    competition.public_dir.mkdir(parents=True)
    competition.private_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    ref, query = rng.normal(size=(2, LENGTH)).cumsum(axis=1)
    np.save(competition.public_dir / "ref_data.npy", ref)
    np.save(competition.public_dir / "query_data.npy", query)
    self_join = brute_force_stomp(ref, None, WINDOW)
    join = brute_force_stomp(ref, query, WINDOW)
    competition.answers.write_text(_r_csv(self_join, join))
    return data_dir, ref, query, self_join, join


def _grade_stomp(data_dir, tmp_path, source: str):
    submission = tmp_path / "submission.py"
    submission.write_text(source)
    return grade_code(submission, Registry(data_dir).get_competition(STOMP))


def _source_of_brute_force() -> str:
    names = [_windows, _distances, _first_min, brute_force_stomp]
    head = f"import numpy as np\n\nWINDOW = {WINDOW}\nEPS = np.finfo(float).eps ** 0.5\n\n"
    body = "\n\n".join(inspect.getsource(f) for f in names)
    return head + body + "\n\nstomp = brute_force_stomp\n"


def _pasted_outputs(self_join, join) -> str:
    """A submission that returns stored outputs for the public inputs."""

    def literal(arrays):
        items = ("None" if a is None else f"np.array({a.tolist()!r})" for a in arrays)
        return "(" + ", ".join(items) + ",)"

    return (
        "import numpy as np\n"
        "inf = np.inf\n"
        f"SELF = {literal(self_join)}\n"
        f"JOIN = {literal(join)}\n"
        "def stomp(ref_data, query_data, window_size, exclusion_zone=0.5, verbose=2):\n"
        "    return JOIN if query_data is not None else SELF\n"
    )


def test_stomp_variant_outputs_follow_from_the_public_outputs(stomp_data):
    hidden_variant = _module(STOMP, "hidden_variant")
    _, ref, query, self_join, join = stomp_data
    got_self = np.column_stack(brute_force_stomp(hidden_variant.transform(ref), None, WINDOW))
    want_self = hidden_variant.expected_self_join(np.column_stack(self_join))
    assert np.isinf(want_self).any()  # never-updated entries (Inf, -Inf) are covered
    np.testing.assert_allclose(got_self, want_self, rtol=1e-9, atol=1e-9)
    variant_join = brute_force_stomp(
        hidden_variant.transform(ref), hidden_variant.transform(query), WINDOW
    )
    want_join = hidden_variant.expected_join(np.column_stack(join[:2]), LENGTH - WINDOW + 1)
    np.testing.assert_allclose(np.column_stack(variant_join[:2]), want_join, rtol=1e-9, atol=1e-9)


def test_stomp_a_working_implementation_passes_both_tests(stomp_data, tmp_path):
    report = _grade_stomp(stomp_data[0], tmp_path, _source_of_brute_force())
    assert report.valid_submission and report.test_metric == "2/2"
    assert report.executed_functions == "1/1"


def test_stomp_outputs_pasted_for_the_public_inputs_fail(stomp_data, tmp_path):
    data_dir, _, _, self_join, join = stomp_data
    report = _grade_stomp(data_dir, tmp_path, _pasted_outputs(self_join, join))
    assert report.valid_submission and report.test_metric == "0/2"
    assert report.executed_functions == "1/1"  # right types and shapes


def test_stomp_sample_like_submission_scores_zero(stomp_data, tmp_path):
    source = "def stomp(ref_data, query_data, window_size, exclusion_zone=0.5, verbose=2):\n"
    source += "    pass\n"
    report = _grade_stomp(stomp_data[0], tmp_path, source)
    assert report.valid_submission and report.test_metric == "0/2"
