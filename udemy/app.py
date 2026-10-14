import gradio as gr
from dotenv import load_dotenv

from naive_implementation.answer import answer_question

load_dotenv(override=True)


def get_text_content(content):
    """Convert Gradio message content into a plain string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return ""


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"

    for doc in context:
        source = doc.metadata.get("source", "Unknown")
        result += (
            f"<span style='color: #ff7800;'>Source: {source}</span>\n\n"
        )
        result += doc.page_content + "\n\n"

    return result


def chat(history):
    if not history:
        return history, "*No conversation yet.*"

    last_message = get_text_content(history[-1].get("content", ""))
    prior = [
        {
            **message,
            "content": get_text_content(message.get("content", "")),
        }
        for message in history[:-1]
    ]

    answer, context = answer_question(last_message, prior)

    updated_history = history + [
        {"role": "assistant", "content": answer}
    ]

    return updated_history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        history = history or []
        return "", history + [
            {"role": "user", "content": message}
        ]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown(
            "# 🏢 Insurellm Expert Assistant\n"
            "Ask me anything about Insurellm!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                    # type="messages",
                )

                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_markdown],
        )

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()