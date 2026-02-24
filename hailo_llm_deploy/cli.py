"""CLI entry point using Typer."""

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from .config import PipelineConfig

app = typer.Typer(
    name="hailo-llm-deploy",
    help="HuggingFace LLM fine-tuning and edge deployment pipeline.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def finetune(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fine-tune a model with LoRA using the given config."""
    _setup_logging(verbose)
    from .finetune import FineTuner
    cfg = PipelineConfig.from_yaml(config)
    trainer = FineTuner(cfg)
    trainer.run()


@app.command()
def export(
    model: Path = typer.Option(..., "--model", "-m", help="Path to merged model directory"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to output directory"),
    dtype: str = typer.Option("float16", "--dtype", help="Export dtype"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Export a model to ONNX format."""
    _setup_logging(verbose)
    from .export import ModelExporter
    exporter = ModelExporter(model_path=model, output_dir=output, dtype=dtype)
    exporter.export()


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    trials: list[str] = typer.Option(None, "--trial", "-t", help="Trial name(s) to evaluate"),
    merged_dir: Path = typer.Option(None, "--merged-dir", help="Directory containing merged models"),
    results_dir: Path = typer.Option(Path("./results"), "--results-dir", help="Results output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate fine-tuned models with ROUGE-L, BERTScore, and LLM Judge."""
    _setup_logging(verbose)
    from .evaluate import Evaluator
    cfg = PipelineConfig.from_yaml(config)
    evaluator = Evaluator(cfg)
    evaluator.load_test_data()

    if merged_dir is None:
        merged_dir = Path("./models/merged")

    if trials is None:
        trials = evaluator.discover_trials(merged_dir)

    if not trials:
        typer.echo("No trial models found.")
        raise typer.Exit(1)

    all_results = []
    for trial_name in trials:
        model_path = merged_dir / trial_name
        result = evaluator.evaluate_trial(trial_name, model_path, results_dir)
        all_results.append(result)

    evaluator.print_results(all_results)


@app.command()
def compile(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    har: Path = typer.Option(None, "--har", help="Path to pre-optimized HAR file"),
    alls: Path = typer.Option(None, "--alls", help="Path to ALLS model script"),
    lora: Path = typer.Option(None, "--lora", help="Path to LoRA safetensors weights"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    output_name: str = typer.Option(None, "--output-name", help="Base name for output files"),
    force: bool = typer.Option(False, "--force", help="Skip HAR availability warning"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Compile LoRA adapter with pre-optimized HAR for Hailo NPU."""
    _setup_logging(verbose)

    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    if not force:
        console.print(Panel(
            "[bold]Hailo GenAI HAR files are not yet publicly available.[/bold]\n\n"
            "The GenAI Model Zoo distributes HEF (compiled binary) only.\n"
            "LoRA compilation requires pre-optimized HAR files, which Hailo\n"
            "has not released to end-users as of 2026-02.\n\n"
            "To proceed, you need:\n"
            "  1. Contact Hailo for enterprise/partner HAR access, or\n"
            "  2. Wait for public release of GenAI LoRA tooling\n\n"
            "Use --force to skip this warning.\n"
            "See: docs/hailo-lora-guide.md for details.",
            title="[yellow]Blocked: HAR Files Unavailable[/yellow]",
            border_style="yellow",
        ))
        raise typer.Exit(1)

    cfg = PipelineConfig.from_yaml(config)

    if har:
        cfg.compile.har_path = har
    if alls:
        cfg.compile.alls_path = alls
    if lora:
        cfg.compile.lora_weights_path = lora
    if output_dir:
        cfg.compile.output_dir = output_dir
    if output_name:
        cfg.compile.output_name = output_name

    if not cfg.compile.har_path:
        console.print(Panel(
            "HAR file path is required. Provide it via:\n"
            "  --har flag, or\n"
            "  compile.har_path in YAML config\n\n"
            "Download pre-optimized HARs from the Hailo GenAI Model Zoo.",
            title="[red]Missing HAR Path[/red]",
            border_style="red",
        ))
        raise typer.Exit(1)

    if not cfg.compile.alls_path:
        console.print(Panel(
            "ALLS model script path is required. Provide it via:\n"
            "  --alls flag, or\n"
            "  compile.alls_path in YAML config\n\n"
            "Download ALLS files from the Hailo GenAI Model Zoo.",
            title="[red]Missing ALLS Path[/red]",
            border_style="red",
        ))
        raise typer.Exit(1)

    try:
        from .compile import HailoCompiler
    except ImportError as e:
        console.print(Panel(
            str(e),
            title="[red]Missing Dependency[/red]",
            border_style="red",
        ))
        raise typer.Exit(1)

    compiler = HailoCompiler(cfg)
    hef_path = compiler.run()

    console.print(Panel(
        f"HEF: {hef_path}\n"
        f"HAR: {cfg.compile.output_dir / f'{cfg.compile.output_name}.compiled.har'}",
        title="[green]Compilation Complete[/green]",
        border_style="green",
    ))


if __name__ == "__main__":
    app()
