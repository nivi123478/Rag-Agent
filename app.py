"""Gradio UI for the Document Q&A agent."""
import gradio as gr

from agent import build_graph
from ingest import Index


def upload(files, idx):
    idx = idx or Index()
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    n = idx.add_files(paths)
    return idx, f"Indexed {n} new chunks ({len(idx.chunks)} total)."


def respond(message, history, idx):
    history = history or []
    if idx is None or not idx.chunks:
        reply, sources = "Please upload and index a document first.", ""
    else:
        try:
            out = build_graph(idx).invoke({"question": message, "history": history, "attempts": 0})
            reply = out["answer"]
            sources = "\n".join(f"- {s}" for s in out.get("sources", []))
        except Exception as e:  # rate limits, bad keys, etc.
            reply, sources = f"Error: {e}", ""
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history, sources


with gr.Blocks(title="Document Q&A Agent") as demo:
    gr.Markdown(
        "# Document Q&A Agent\nUpload PDFs or text files, then ask questions. "
        "Answers cite their sources. Don't upload sensitive files: free LLM tiers may use prompts for training."
    )
    idx_state = gr.State(None)
    with gr.Row():
        files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"], label="Documents")
        status = gr.Markdown("No documents indexed yet.")
    index_btn = gr.Button("Index documents")
    chat = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Ask a question about your documents...", label="Question")
    sources_box = gr.Markdown(label="Sources")

    index_btn.click(upload, [files, idx_state], [idx_state, status])
    msg.submit(respond, [msg, chat, idx_state], [msg, chat, sources_box])

if __name__ == "__main__":
    demo.launch()
