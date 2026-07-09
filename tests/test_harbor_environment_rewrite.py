import importlib
import sys
import types


def _stub_harbor_modules(monkeypatch):
    module_names = [
        "harbor",
        "harbor.environments",
        "harbor.environments.base",
        "harbor.environments.docker",
        "harbor.environments.docker.docker",
        "harbor.environments.factory",
        "harbor.models",
        "harbor.models.environment_type",
        "harbor.models.task",
        "harbor.models.task.config",
        "harbor.models.task.task",
        "harbor.models.trial",
        "harbor.models.trial.paths",
        "harbor.verifier",
        "harbor.verifier.verifier",
    ]
    modules = {name: types.ModuleType(name) for name in module_names}

    modules["harbor.environments.base"].BaseEnvironment = object
    modules["harbor.environments.base"].ExecResult = object

    class DockerEnvironment:
        _image_build_locks = {}

    modules["harbor.environments.docker.docker"].DockerEnvironment = DockerEnvironment

    class EnvironmentFactory:
        pass

    modules["harbor.environments.factory"].EnvironmentFactory = EnvironmentFactory

    class EnvironmentType:
        DOCKER = "docker"

    modules["harbor.models.environment_type"].EnvironmentType = EnvironmentType
    modules["harbor.models.task.config"].EnvironmentConfig = object
    modules["harbor.models.task.task"].Task = object
    modules["harbor.models.trial.paths"].TrialPaths = object
    modules["harbor.verifier.verifier"].Verifier = object

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _import_harbor_environment(monkeypatch):
    _stub_harbor_modules(monkeypatch)
    sys.modules.pop("echo_rl.terminal_agent.harbor_environment", None)
    return importlib.import_module("echo_rl.terminal_agent.harbor_environment")


def test_rewrite_pytest_install_in_dockerfile_run(monkeypatch):
    harbor_environment = _import_harbor_environment(monkeypatch)

    dockerfile = (
        "FROM ubuntu:22.04\n"
        "RUN set -e && \\\n"
        "    apt-get update -y && \\\n"
        "    pip3 install --no-cache-dir pytest && \\\n"
        "    echo done\n"
    )

    rewritten = harbor_environment._rewrite_pytest_installs(
        dockerfile,
        "pip3 install --timeout 60 --retries 5 -i https://mirror.example/simple pytest",
    )

    assert "pip3 install --no-cache-dir pytest" not in rewritten
    assert "pip3 install --timeout 60 --retries 5 -i https://mirror.example/simple pytest" in rewritten
    assert "echo done" in rewritten


def test_inject_docker_wheelhouse_rewrites_dockerfile_mirror(monkeypatch, tmp_path):
    harbor_environment = _import_harbor_environment(monkeypatch)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:22.04\n"
        "RUN python3 -m pip install --no-cache-dir pytest && echo ok\n"
    )

    monkeypatch.setenv("ECHO_DOCKER_PIP_MODE", "mirror")
    monkeypatch.setenv("ECHO_DOCKER_PIP_INDEX_URL", "https://mirror.example/simple")

    harbor_environment._inject_docker_wheelhouse(tmp_path)

    rewritten = dockerfile.read_text()
    assert "python3 -m pip install --no-cache-dir pytest" not in rewritten
    assert (
        "python3 -m pip install --timeout 60 --retries 5 "
        "-i https://mirror.example/simple pytest"
    ) in rewritten
