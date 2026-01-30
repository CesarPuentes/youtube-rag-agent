# YouTube AI Agent (LangGraph)

This project implements an intelligent agent capable of interacting with YouTube to search for videos, extract metadata, and summarize content using transcripts. It uses **LangGraph** to manage the Reasoning + Acting (ReAct) loop.

## Features

-   **Automated ReAct Loop**: Uses LangGraph's `create_react_agent` to handle decision-making logic.
-   **Ollama Integration**: Runs local models such as `qwen2.5:7b`.
-   **YouTube Tools**:
    -   Video ID extraction.
    -   Video search.
    -   Full metadata retrieval (views, duration, channel, etc.).
    -   Transcript downloading for summarization.

## Prerequisites

1.  **Ollama**: You must have Ollama installed and the `qwen2.5:7b` model downloaded.
    ```bash
    ollama pull qwen2.5:7b
    ```
2.  **Python Environment**: Using a virtual environment is recommended.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

To run the agent with the predefined example:

```bash
python youtube_agent_langgraph.py
```

### Code Example

```python
from youtube_agent_langgraph import agent

query = "Search for a video about LangGraph and summarize it"
response = agent(query)
print(response)
```

## Project Structure

-   `youtube_agent.py`: Version with a manually implemented ReAct loop (for educational purposes).
-   `youtube_agent_langgraph.py`: Optimized version using LangGraph abstraction.
-   `requirements.txt`: Project dependencies.
