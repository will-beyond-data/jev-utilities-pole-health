"""The `polehealth` command line."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import httpx

from . import evaluate, images, serve, sources
from .demo_data import build_demo_data, decide_pole
from .jev import Backend, BackendError, make_backend
from .questions import DEFAULT_POLICY, ENGINEER_THRESHOLD, v1_request
from .register import DEFAULT_SEED, build_register, load_register, load_wrc, write_register

BACKENDS = click.Choice(["http", "space", "mock"])


def echo(message: str) -> None:
    click.echo(message, err=True)


def backend_options(function: Callable[..., Any]) -> Callable[..., Any]:
    @click.option(
        "--backend", "backend_name", type=BACKENDS, help="Where the model runs: http (self-hosted), space, or mock."
    )
    @click.option(
        "--base-url",
        envvar="MJ_URL",
        default=None,
        help="Model server URL for http [env MJ_URL, default http://127.0.0.1:8083].",
    )
    @click.option("--api-key", envvar="MJ_API_KEY", default=None, help="Bearer key for http [env MJ_API_KEY].")
    @click.option(
        "--hf-token", envvar="HF_TOKEN", default=None, help="Hugging Face token for the space backend [env HF_TOKEN]."
    )
    @click.option(
        "--hardware",
        default=None,
        help="Hardware label to record, for example '1x NVIDIA A100 80GB'. Defaults to what the server reports.",
    )
    @click.option(
        "--seed", type=int, default=DEFAULT_SEED, show_default=True, help="Seed for the mock backend and for sampling."
    )
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    return wrapper


def open_backend(
    name: str | None,
    base_url: str | None,
    api_key: str | None,
    hf_token: str | None,
    hardware: str | None,
    seed: int,
    check: bool = True,
) -> tuple[Backend, str]:
    if name is None:
        raise click.UsageError("choose a backend with --backend {http,space,mock}")
    try:
        backend = make_backend(
            name, base_url=base_url, api_key=api_key, hf_token=hf_token, seed=seed, hardware=hardware
        )
    except BackendError as error:
        raise click.ClickException(str(error)) from error
    if name == "http" and check:
        try:
            health = backend.health()  # type: ignore[attr-defined]
        except (httpx.HTTPError, ValueError) as error:
            raise click.ClickException(f"cannot reach the model server at {backend.base_url}: {error}") from error  # type: ignore[attr-defined]
        if health.get("status") != "ready":
            raise click.ClickException(f"model server is not ready yet: {health}")
    return backend, hardware or backend.describe_hardware()


def mock_banner(backend: Backend) -> None:
    if backend.name == "mock":
        echo("MOCK BACKEND: these answers are synthetic and are not model results.")


@click.group()
@click.version_option()
def main() -> None:
    """Triage electricity pole health with Matilda Jev: photo to health, photo plus record to action."""


@main.command()
@click.option("--only", default=None, help=f"Comma-separated subset of: {','.join(sources.SOURCE_KEYS)}.")
@click.option("--limit", type=int, default=None, help="Cap image downloads per image source (balanced across labels).")
def fetch(only: str | None, limit: int | None) -> None:
    """Download source data into data/raw/ (safe to re-run)."""
    keys = [k.strip() for k in only.split(",") if k.strip()] if only else None
    try:
        failed = sources.fetch(keys, limit, sources.raw_dir(), echo)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    if failed:
        raise click.ClickException(f"failed: {', '.join(failed)}. Re-run to retry; finished downloads are kept.")
    echo("done")


@main.command("build-register")
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output file [default data/register.json].")
def build_register_command(seed: int, out: Path | None) -> None:
    """Pair OSM poles with wrc50 inspection histories and seeded network attributes."""
    root = sources.repo_root()
    raw = sources.raw_dir(root)
    osm, wrc = sources.osm_path(raw), sources.wrc_path(raw, "wrc50")
    for path, key in ((osm, "osm"), (wrc, "wrc50")):
        if not path.is_file():
            raise click.ClickException(f"{path} is missing; run `polehealth fetch --only {key}`")
    register = build_register(json.loads(osm.read_text(encoding="utf-8")), load_wrc(wrc), seed=seed)
    target = out or root / "data" / "register.json"
    write_register(register, target)
    zones: dict[str, int] = {}
    for pole in register["poles"]:
        zones[pole["record"]["bushfire_zone"]] = zones.get(pole["record"]["bushfire_zone"], 0) + 1
    echo(f"wrote {target} with {len(register['poles'])} poles (bushfire zones {zones})")


def _print_v1(answers: dict[str, Any]) -> None:
    for key, answer in answers.items():
        if answer["type"] == "choice":
            top = answer["probabilities"][answer["choice"]]
            click.echo(f"  {key:<11} {answer['choice']:<14} p={top:.2f}")
        elif answer["type"] == "score":
            levels = {i: v for i, v in answer["probabilities"].items()}
            best = max(levels, key=levels.__getitem__)
            click.echo(f"  {key:<11} score {answer['score']:.2f} of 4 (most likely level {best}, p={levels[best]:.2f})")
        else:
            click.echo(f"  {key:<11} p(true)={answer['noul']:.2f}")


@main.command()
@backend_options
@click.option("--image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--record-id", default=None, help="Pole id in data/register.json (osm-123 or 123); adds the v2 action.")
@click.option("--policy-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def ask(
    backend_name,
    base_url,
    api_key,
    hf_token,
    hardware,
    seed,
    image: Path,
    record_id: str | None,
    policy_file: Path | None,
) -> None:
    """Ask the model about one photo (v1), and with --record-id about the pole record too (v2)."""
    backend, _ = open_backend(backend_name, base_url, api_key, hf_token, hardware, seed)
    mock_banner(backend)
    record = None
    if record_id:
        register = load_register(sources.repo_root() / "data" / "register.json")
        wanted = record_id if record_id.startswith("osm-") else f"osm-{record_id}"
        match = next((p for p in register["poles"] if p["id"] == wanted), None)
        if match is None:
            raise click.ClickException(f"{wanted} is not in data/register.json")
        record = match["record"]
    policy = policy_file.read_text(encoding="utf-8").strip() if policy_file else DEFAULT_POLICY
    try:
        if record is None:
            result = backend.decide(v1_request(images.data_url(image), backend.model))
            click.echo(f"v1 photo to health ({result.latency_ms:.0f} ms wall, model {_ms(result.model_ms)})")
            _print_v1(result.response["answers"])
        else:
            entry = decide_pole(backend, images.data_url(image), record, policy, ENGINEER_THRESHOLD)
            click.echo(
                f"v1 photo to health ({entry['v1']['latency_ms']} ms wall, model {_ms(entry['v1']['model_ms'])})"
            )
            _print_v1(entry["v1"]["answers"])
            click.echo(
                f"v2 photo plus record to action ({entry['v2']['latency_ms']} ms wall, "
                f"model {_ms(entry['v2']['model_ms'])})"
            )
            for key, probability in entry["v2"]["answers"]["action"]["probabilities"].items():
                click.echo(f"  {key:<10} {probability:.2f}")
            click.echo(f"  safety_risk_now p(true)={entry['v2']['answers']['safety_risk_now']['noul']:.2f}")
            click.echo(f"action: {entry['action']} (confidence {entry['confidence']}), route: {entry['route']}")
    except BackendError as error:
        raise click.ClickException(str(error)) from error
    finally:
        backend.close()


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


@main.command("eval")
@backend_options
@click.option("--task", type=click.Choice([*evaluate.TASKS, "all"]), default="all", show_default=True)
@click.option(
    "--limit", type=int, default=None, help="Items per task (balanced across labels). Default: everything downloaded."
)
@click.option(
    "--concurrency", type=int, default=1, show_default=True, help="Parallel requests. Keep 1 when measuring latency."
)
@click.option(
    "--resume",
    "resume_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Continue an existing run directory; items already in results.jsonl are skipped.",
)
def eval_command(
    backend_name,
    base_url,
    api_key,
    hf_token,
    hardware,
    seed,
    task: str,
    limit: int | None,
    concurrency: int,
    resume_dir: Path | None,
) -> None:
    """Score the model on labelled sets and write runs/<timestamp>-<backend>/."""
    backend, hardware_label = open_backend(backend_name, base_url, api_key, hf_token, hardware, seed)
    mock_banner(backend)
    root = sources.repo_root()
    run_dir = resume_dir or evaluate.new_run_dir(root / "runs", backend.name)
    tasks = list(evaluate.TASKS) if task == "all" else [task]
    echo(f"run directory: {run_dir}")
    try:
        summary = evaluate.run_eval(
            backend,
            tasks,
            sources.raw_dir(root),
            run_dir,
            limit,
            concurrency,
            hardware_label,
            seed,
            base_url if backend.name == "http" else None,
            echo,
        )
    finally:
        backend.close()
    click.echo(f"{'task':<22}{'n':>6}{'accuracy':>10}{'macro_f1':>10}{'ece':>8}{'wall p50':>10}{'model p50':>11}")
    for name in evaluate.TASKS:
        if name in summary["tasks"]:
            row = summary["tasks"][name]
            model = row["model_ms"]["p50"]
            click.echo(
                f"{name:<22}{row['n']:>6}{row['accuracy']:>10.3f}{row['macro_f1']:>10.3f}{row['ece']:>8.3f}"
                f"{row['latency_ms']['p50']:>10.1f}{'n/a' if model is None else format(model, '.1f'):>11}"
            )
    click.echo(f"wrote {run_dir / 'results.jsonl'} and {run_dir / 'eval_summary.json'}")


@main.command("demo-data")
@backend_options
@click.option(
    "--run",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Run directory whose eval_summary.json is shown and where decisions are cached. "
    "Default: latest run for the backend.",
)
@click.option(
    "--inspect",
    "inspect_n",
    type=int,
    default=400,
    show_default=True,
    help="How many poles get a photo and a decision.",
)
@click.option(
    "--policy-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Plain-English maintenance policy for v2. Changing it re-runs the v2 decisions.",
)
@click.option("--concurrency", type=int, default=1, show_default=True)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory [default demo/data].",
)
def demo_data_command(
    backend_name,
    base_url,
    api_key,
    hf_token,
    hardware,
    seed,
    run_dir: Path | None,
    inspect_n: int,
    policy_file: Path | None,
    concurrency: int,
    out: Path | None,
) -> None:
    """Build demo/data/*.json and resized photos from the register and a model backend."""
    root = sources.repo_root()
    if backend_name is None and run_dir is not None:
        backend_name = json.loads((run_dir / "run.json").read_text())["backend"]
    backend, hardware_label = open_backend(backend_name, base_url, api_key, hf_token, hardware, seed)
    mock_banner(backend)
    register_path = root / "data" / "register.json"
    if not register_path.is_file():
        raise click.ClickException("data/register.json is missing; run `polehealth build-register`")
    if run_dir is None:
        run_dir = evaluate.latest_run(root / "runs", backend.name) or evaluate.new_run_dir(root / "runs", backend.name)
        echo(f"run directory: {run_dir}")
    if not (run_dir / "run.json").is_file():
        evaluate.write_run_meta(
            run_dir, backend, [], hardware_label, None, seed, base_url if backend.name == "http" else None
        )
    recorded = json.loads((run_dir / "run.json").read_text())["backend"]
    if recorded != backend.name:
        raise click.ClickException(
            f"{run_dir} holds a {recorded} run but the backend is {backend.name}; do not mix them"
        )
    policy = policy_file.read_text(encoding="utf-8").strip() if policy_file else DEFAULT_POLICY
    out_dir = out or root / "demo" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        built = build_demo_data(
            load_register(register_path),
            sources.raw_dir(root),
            out_dir,
            run_dir,
            backend,
            hardware_label,
            inspect_n=inspect_n,
            policy=policy,
            seed=seed,
            concurrency=concurrency,
            log=echo,
        )
    except (BackendError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        backend.close()
    decisions = built["decisions"]["decisions"]
    routes = [d["route"] for d in decisions.values()]
    actions: dict[str, int] = {}
    for d in decisions.values():
        actions[d["action"]] = actions.get(d["action"], 0) + 1
    click.echo(
        f"wrote {out_dir} (backend {backend.name}{', MOCK' if backend.name == 'mock' else ''}): "
        f"{len(decisions)} decisions {actions}, engineer queue {routes.count('engineer')}"
    )


@main.command()
@click.option("--gpu-hourly-usd", type=float, required=True, help="What you paid per GPU hour, in USD.")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--aud-per-usd", type=float, default=1.52, show_default=True)
def cost(gpu_hourly_usd: float, run_dir: Path, aud_per_usd: float) -> None:
    """Cost per 1,000 poles from the measured model time in a run, saved into its eval_summary.json."""
    try:
        result = evaluate.compute_cost(run_dir, gpu_hourly_usd, aud_per_usd)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"basis: {result['basis']}")
    click.echo(f"mean model time per pole: {result['mean_model_ms_per_pole']} ms over {result['n_poles']} poles")
    click.echo(f"decisions per hour: {result['decisions_per_hour']:,}")
    click.echo(f"cost per 1,000 poles: USD {result['usd_per_1000_poles']:.4f} / AUD {result['aud_per_1000_poles']:.4f}")
    click.echo(result["note"])


@main.command("serve")
@backend_options
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--host", default="127.0.0.1", show_default=True, help="Bind address. Use 0.0.0.0 only on a trusted network."
)
@click.option("--policy-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def serve_command(
    backend_name, base_url, api_key, hf_token, hardware, seed, port: int, host: str, policy_file: Path | None
) -> None:
    """Serve demo/ plus /api/health and /api/decide for the live drop-a-photo moment."""
    backend, _ = open_backend(backend_name, base_url, api_key, hf_token, hardware, seed, check=False)
    mock_banner(backend)
    demo_dir = sources.repo_root() / "demo"
    if not demo_dir.is_dir():
        raise click.ClickException(f"{demo_dir} does not exist")
    policy = policy_file.read_text(encoding="utf-8").strip() if policy_file else DEFAULT_POLICY
    state = serve.ServeState(backend, policy)
    server = serve.make_server(state, demo_dir, host, port)
    echo(f"serving {demo_dir} at http://{host}:{port}/ with the {backend.name} backend (Ctrl-C to stop)")
    if not state.ready():
        echo("the model server is not ready yet; /api/health will answer 503 until it is")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        echo("stopping")
    finally:
        server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
