import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from typing import Optional

from rich.console import Console
from rich.panel import Panel

from app.core.config import settings
from app.core.logging import logger
from app.ingestion.processor import IngestionProcessor

console = Console()


def run_ingestion(
    source_dir: Optional[str] = None,
    file_path: Optional[str] = None,
    strategy: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Main entry point for document ingestion.
    Processes specified file or directory and builds the vector index.
    """
    target_raw_dir = source_dir or str(settings.paths.raw_data_dir)
    target_output_path = output_path or settings.vector_db.faiss_path

    console.print(
        Panel.fit(
            "[bold cyan]RAGVerse Ingestion Pipeline[/bold cyan]\n"
            f"[yellow]Source Dir:[/yellow] {target_raw_dir}\n"
            f"[yellow]Target Vector Store:[/yellow] {target_output_path}\n"
            f"[yellow]Strategy Override:[/yellow] {strategy or 'Auto-select'}",
            title="Document Ingestion Engine",
            border_style="cyan",
        )
    )

    processor = IngestionProcessor(output_index_path=target_output_path)

    if file_path:
        chunks = processor.process_file(file_path, strategy_name=strategy)
    else:
        chunks = processor.process_directory(target_raw_dir, strategy_name=strategy)

    if chunks:
        processor.build_and_save_index(chunks)
        console.print(
            Panel(
                f"[bold green]Ingestion complete![/bold green] Processed [cyan]{len(chunks)}[/cyan] total chunks.",
                title="Success",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold yellow]Ingestion finished with 0 chunks generated.[/bold yellow] Please check source files.",
                title="Warning",
                border_style="yellow",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGVerse Document Ingestion Engine CLI")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(settings.paths.raw_data_dir),
        help=f"Path to raw documents directory (default: {settings.paths.raw_data_dir})",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to a single document file to process",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["recursive", "markdown", "code", "token", "character"],
        help="Explicit chunking strategy override (default: Auto-select)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=settings.vector_db.faiss_path,
        help=f"Path to output vector index (default: {settings.vector_db.faiss_path})",
    )

    args = parser.parse_args()

    try:
        run_ingestion(
            source_dir=args.source_dir,
            file_path=args.file_path,
            strategy=args.strategy,
            output_path=args.output_path,
        )
    except Exception as error:
        logger.error(f"Ingestion CLI failed: {error}", exc_info=True)
        console.print(Panel(str(error), title="Ingestion Error", border_style="red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
