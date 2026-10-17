import shutil
from pathlib import Path

import numpy as np

from timeseriesgym.utils import clone_repository

TODO = "...#TODO: fill in the missing part"
# The four lines of `DataEmbedding_inverted` that models/iTransformer.py marks TODO, as the
# clone's layers/Embed.py has them filled in, and the TODOs that replace them there.
EMBED_ANSWERS = [
    (
        "        self.value_embedding = nn.Linear(c_in, d_model)\n"
        "        self.dropout = nn.Dropout(p=dropout)\n",
        f"        self.value_embedding = {TODO}\n        self.dropout = {TODO}\n",
    ),
    (
        "            x = self.value_embedding(x)\n"
        "        else:\n"
        "            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))\n",
        f"            x = {TODO}\n        else:\n            x = {TODO}\n",
    ),
]


def stub_embedding(embed: Path) -> None:
    """Replace the answer to the task's TODOs in layers/Embed.py with the same TODOs.

    The clone's layers/Embed.py has `DataEmbedding_inverted` filled in, which is exactly the
    class that models/iTransformer.py asks the agent to complete. The class is stubbed rather
    than deleted, because models/TimeXer.py imports it. Afterwards it reads as it does in
    models/iTransformer.py.
    """
    text = embed.read_text()
    for answer, todo in EMBED_ANSWERS:
        if text.count(answer) != 1:
            raise ValueError(f"{embed}: expected the filled-in DataEmbedding_inverted once")
        text = text.replace(answer, todo)
    embed.write_text(text)


def prepare(raw: Path, public: Path, private: Path):
    solution = raw / "pred.npy"
    solution = np.load(solution)
    sample_submission = np.zeros(solution.shape)

    np.save(private / "test.npy", solution)
    np.save(public / "sample_submission.npy", sample_submission)

    # Download the tsl GitHub repository
    clone_repository(
        "https://github.com/raycai420/Time-Series-Library.git",
        public / "Time-Series-Library",
        branch="itransformer-weather-eval",
        commit="493256f1584e126ca2d4c3cf93d0a4a339791153",
    )

    # Remove all .json and .csv files from the cloned repository
    # (the path previously used lowercase "time-series-library", so nothing was removed)
    for file in (public / "Time-Series-Library").rglob("*"):
        if file.suffix in [".json", ".csv"]:
            file.unlink()

    # layers/Embed.py ships the answer to the TODOs of models/iTransformer.py
    stub_embedding(public / "Time-Series-Library" / "layers" / "Embed.py")

    # move weather data
    weather_data = raw / "weather.csv"
    weather_data.rename(public / "weather.csv")

    # the description points the agent to the excerpt of the iTransformer paper
    shutil.copy(raw / "itransformer.txt", public / "itransformer.txt")
