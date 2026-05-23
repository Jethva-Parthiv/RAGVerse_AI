from uuid import uuid4
from time import sleep

from rich.markdown import Markdown
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from app.services.chat_service import get_chat_response


console = Console()


def typewriter_print(text: str, delay: float = 0.01):
    """
    Print text with typing animation.
    """
    for char in text:
        console.print(char, end="")
        sleep(delay)

    console.print()


def main():
    thread_id = f"chat-{uuid4().hex[:8]}"

    console.print(
        Panel.fit(
            f"[bold cyan]RAG Chatbot Started[/bold cyan]\n"
            f"[yellow]Thread ID:[/yellow] {thread_id}\n\n"
            f"Type [green]'exit'[/green] or [green]'bye'[/green] to quit.",
            title="AI Assistant",
            border_style="blue"
        )
    )

    while True:
        query = console.input(
            "\n[bold green]User > [/bold green]"
        ).strip()

        if not query:
            console.print(
                "[red]Please enter a question.[/red]"
            )
            continue

        if query.lower() in ["exit", "bye"]:
            console.print(
                "\n[bold cyan]AI >[/bold cyan] Bye 👋"
            )
            break

        try:
            console.print(
                "\n[yellow]Thinking...[/yellow]"
            )

            response = get_chat_response(
                query=query,
                thread_id=thread_id
            )

            ai_text = Text()
            ai_text.append("AI > ", style="bold cyan")
            ai_text.append(response)


            console.print(
                Panel(
                    Markdown(response),
                    border_style="cyan",
                    title="AI Response"
                )
            )

        except Exception as error:
            console.print(
                Panel(
                    str(error),
                    title="Error",
                    border_style="red"
                )
            )


if __name__ == "__main__":
    main()