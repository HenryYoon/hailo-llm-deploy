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


if __name__ == "__main__":
    app()
