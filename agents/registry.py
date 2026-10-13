from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agents.model_proxy import parse_config as parse_model_proxy_config
from agents.utils import is_env_var
from timeseriesgym.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    agents_dir: Path
    start: Path
    dockerfile: Path
    kwargs: dict
    env_vars: dict
    privileged: bool = False
    kwargs_type: str | None = None
    # the parsed `model_proxy` config (agents/model_proxy.py); it holds the API keys
    model_proxy: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        assert isinstance(self.start, Path), "Agent start script must be a pathlib.Path object."
        assert isinstance(self.dockerfile, Path), "Agent dockerfile must be a pathlib.Path object."
        assert isinstance(self.kwargs, dict), "Agent kwargs must be a dictionary."
        assert isinstance(self.privileged, bool), "Agent privileged must be a boolean."

        if self.kwargs_type is not None:
            assert isinstance(self.kwargs_type, str), "Agent kwargs_type must be a string."
        else:  # i.e., self.kwargs_type is None
            assert self.kwargs == {}, "Agent kwargs_type must be set if kwargs are provided."

        assert isinstance(self.env_vars, dict), "Agent env_vars must be a dictionary."

        assert self.start.exists(), f"start script {self.start} does not exist."
        assert self.dockerfile.exists(), f"dockerfile {self.dockerfile} does not exist."

    @staticmethod
    def from_dict(data: dict) -> "Agent":
        agents_dir = Path(data["agents_dir"])
        try:
            return Agent(
                id=data["id"],
                name=data["name"],
                agents_dir=agents_dir,
                start=agents_dir / data["start"],
                dockerfile=agents_dir / data["dockerfile"],
                kwargs=data.get("kwargs", {}),
                kwargs_type=data.get("kwargs_type", None),
                env_vars=data.get("env_vars", {}),
                privileged=data.get("privileged", False),
                model_proxy=data.get("model_proxy"),
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in agent config!") from e


class Registry:
    def get_agents_dir(self) -> Path:
        """Retrieves the agents directory within the registry."""

        return Path(__file__).parent

    def get_agent(self, agent_id: str) -> Agent:
        """Fetch the agent from the registry."""

        agents_dir = self.get_agents_dir()

        for fpath in agents_dir.glob("**/config.yaml"):
            with open(fpath) as f:
                contents = yaml.safe_load(f)

            if agent_id not in contents:
                continue

            logger.debug(f"Fetching {fpath}")

            kwargs = contents[agent_id].get("kwargs", {})
            kwargs_type = contents[agent_id].get("kwargs_type", None)
            env_vars = contents[agent_id].get("env_vars", {})
            privileged = contents[agent_id].get("privileged", False)

            # kwargs and env_vars reach the container, where agent code can read them: secrets
            # (API keys) go to the model proxy, which keeps them on the host
            for section, values in (("kwargs", kwargs), ("env_vars", env_vars)):
                secret = sorted(k for k, v in values.items() if is_env_var(v))
                if secret:
                    raise ValueError(
                        f"Agent `{agent_id}`: {section} {secret} would put a secret in the agent's "
                        "container; give provider API keys to `model_proxy` instead (see "
                        "agents/README.md)."
                    )
            model_proxy = contents[agent_id].get("model_proxy")
            if model_proxy is not None:
                model_proxy = parse_model_proxy_config(model_proxy)

            return Agent.from_dict(
                {
                    **contents[agent_id],
                    "id": agent_id,
                    "name": fpath.parent.name,
                    "agents_dir": agents_dir,
                    "kwargs": kwargs,
                    "kwargs_type": kwargs_type,
                    "env_vars": env_vars,
                    "privileged": privileged,
                    "model_proxy": model_proxy,
                }
            )

        raise ValueError(f"Agent with id {agent_id} not found")


registry = Registry()
