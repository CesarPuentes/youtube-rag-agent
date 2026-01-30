#!/usr/bin/env python3
"""
YouTube Tools Agent - LangChain with Ollama
============================================

This script demonstrates how to build an AI agent that can interact with YouTube
using custom tools. It shows three different methods of executing tool calls:

1. MANUAL METHOD: Step-by-step tool execution with explicit message handling
2. CHAIN METHOD: Using LangChain's RunnablePassthrough for a fixed tool sequence
3. AGENT LOOP METHOD: A flexible loop that handles any tool-calling sequence

The agent uses Ollama (qwen2.5:7b) as the LLM backend.
"""

import re
import json
import warnings
import logging
from typing import List, Dict

# LangChain imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama

# YouTube-related imports
from pytube import YouTube, Search
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

# =============================================================================
# SETUP: Suppress warnings and configure logging
# =============================================================================

warnings.filterwarnings("ignore")

# Suppress pytube errors
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dlp_logger = logging.getLogger('yt_dlp')
yt_dlp_logger.setLevel(logging.ERROR)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# Each tool is decorated with @tool, which converts it into a LangChain tool
# that can be bound to an LLM. The docstring becomes the tool's description,
# which the LLM uses to understand when to call each tool.

@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    # Regex pattern matches video IDs in various YouTube URL formats:
    # - youtube.com/watch?v=VIDEO_ID
    # - youtu.be/VIDEO_ID
    # - youtube.com/embed/VIDEO_ID
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        # Join all transcript snippets into a single string
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query (str): The search term to look for on YouTube
        
    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}"
            }
            for yt in s.results
        ]
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including title, views, duration, channel, likes, comments, and chapters."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dlp_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'views': info.get('view_count'),
            'duration': info.get('duration'),
            'channel': info.get('uploader'),
            'likes': info.get('like_count'),
            'comments': info.get('comment_count'),
            'chapters': info.get('chapters', [])
        }


@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dlp_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]


# =============================================================================
# LLM SETUP
# =============================================================================

# Create the list of available tools
tools = [extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_thumbnails]

# Tool mapping for executing tools by name
tool_mapping = {
    "get_thumbnails": get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}

# Initialize the LLM with Ollama
# ChatOllama is used because it supports tool calling (vs OllamaLLM which doesn't)
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.5
)

# Bind tools to the LLM - this tells the LLM about available tools
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# HELPER FUNCTION: Execute a tool call
# =============================================================================

def execute_tool(tool_call: dict) -> ToolMessage:
    """
    Execute a single tool call and return a ToolMessage.
    
    The ToolMessage is added to the conversation history so the LLM
    can see the result of its tool call.
    """
    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    except Exception as e:
        return ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id=tool_call["id"]
        )


# =============================================================================
# METHOD 1: MANUAL STEP-BY-STEP EXECUTION
# =============================================================================
# This method shows exactly what happens at each step of the conversation.
# It's verbose but educational - you can see each tool call and response.

def manual_summarize_video(video_url: str) -> str:
    """
    Manually orchestrate tool calls to summarize a YouTube video.
    
    This approach:
    1. Sends initial query to LLM
    2. Executes extract_video_id tool
    3. Sends result back to LLM
    4. LLM calls fetch_transcript
    5. Sends transcript to LLM
    6. LLM generates summary
    
    NOTE: This assumes a specific sequence of tool calls. If the LLM
    decides to call different tools, it may not work as expected.
    """
    print("=== MANUAL METHOD ===\n")
    
    # Step 1: Create initial message
    query = f"I want to summarize youtube video: {video_url} in english"
    messages = [HumanMessage(content=query)]
    print(f"Query: {query}\n")
    
    # Step 2: First LLM call - expects extract_video_id tool call
    response_1 = llm_with_tools.invoke(messages)
    messages.append(response_1)
    print(f"LLM Response 1 - Tool calls: {[tc['name'] for tc in response_1.tool_calls]}")
    
    # Step 3: Execute first tool call
    if response_1.tool_calls:
        tool_call_1 = response_1.tool_calls[0]
        video_id = tool_mapping[tool_call_1['name']].invoke(tool_call_1['args'])
        messages.append(ToolMessage(content=video_id, tool_call_id=tool_call_1['id']))
        print(f"Tool result: {video_id}\n")
    
    # Step 4: Second LLM call - expects fetch_transcript tool call
    response_2 = llm_with_tools.invoke(messages)
    messages.append(response_2)
    print(f"LLM Response 2 - Tool calls: {[tc['name'] for tc in response_2.tool_calls]}")
    
    # Step 5: Execute second tool call
    if response_2.tool_calls:
        tool_call_2 = response_2.tool_calls[0]
        transcript = tool_mapping[tool_call_2['name']].invoke(tool_call_2['args'])
        messages.append(ToolMessage(content=transcript, tool_call_id=tool_call_2['id']))
        print(f"Transcript fetched: {len(transcript)} characters\n")
    
    # Step 6: Final LLM call - generates summary
    summary_response = llm_with_tools.invoke(messages)
    
    return summary_response.content


# =============================================================================
# METHOD 2: LANGCHAIN CHAIN (FIXED SEQUENCE)
# =============================================================================
# This uses LangChain's RunnablePassthrough to create a pipeline.
# More elegant than manual, but still assumes a fixed 2-tool sequence.

summarization_chain = (
    # Start with initial query, wrap in HumanMessage
    RunnablePassthrough.assign(
        messages=lambda x: [HumanMessage(content=x["query"])]
    )
    # First LLM call (expects extract_video_id)
    | RunnablePassthrough.assign(
        ai_response=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process first tool call
    | RunnablePassthrough.assign(
        tool_messages=lambda x: [
            execute_tool(tc) for tc in x["ai_response"].tool_calls
        ]
    )
    # Update message history
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response"]] + x["tool_messages"]
    )
    # Second LLM call (expects fetch_transcript)
    | RunnablePassthrough.assign(
        ai_response2=lambda x: llm_with_tools.invoke(x["messages"])
    )
    # Process second tool call
    | RunnablePassthrough.assign(
        tool_messages2=lambda x: [
            execute_tool(tc) for tc in x["ai_response2"].tool_calls
        ]
    )
    # Final message update
    | RunnablePassthrough.assign(
        messages=lambda x: x["messages"] + [x["ai_response2"]] + x["tool_messages2"]
    )
    # Generate final summary
    | RunnablePassthrough.assign(
        summary=lambda x: llm_with_tools.invoke(x["messages"]).content
    )
    # Return just the summary text
    | RunnableLambda(lambda x: x["summary"])
)


def chain_summarize_video(video_url: str) -> str:
    """
    Use LangChain chain to summarize a video.
    
    NOTE: This chain assumes exactly 2 tool calls in sequence.
    The prompt must guide the LLM to use the expected tools.
    """
    print("=== CHAIN METHOD ===\n")
    
    # Include instructions to guide the LLM to use the expected tools
    result = summarization_chain.invoke({
        "query": f"""Summarize this YouTube video: {video_url}

Use extract_video_id first, then fetch_transcript, then summarize."""
    })
    
    return result


# =============================================================================
# METHOD 3: AGENT LOOP (FLEXIBLE - RECOMMENDED)
# =============================================================================
# This is the most robust approach. It loops until the LLM finishes,
# handling any number of tool calls in any order.

def run_agent_loop(query: str, max_iterations: int = 10, verbose: bool = True) -> str:
    """
    Run an agent loop that handles any sequence of tool calls.
    
    This implements the ReAct (Reasoning + Acting) pattern:
    1. LLM reasons about what to do
    2. LLM acts by calling tools (or generates final answer)
    3. Tool results are added to conversation
    4. Loop continues until LLM generates text without tool calls
    
    Args:
        query: The user's question or request
        max_iterations: Maximum number of LLM calls to prevent infinite loops
        verbose: If True, print progress information
        
    Returns:
        The LLM's final text response
    """
    if verbose:
        print("=== AGENT LOOP METHOD ===\n")
        print(f"Query: {query}\n")
    
    messages = [HumanMessage(content=query)]
    
    for i in range(max_iterations):
        if verbose:
            print(f"--- Iteration {i + 1} ---")
        
        # Get LLM response
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # If no tool calls, we have our final answer
        if not response.tool_calls:
            if verbose:
                print("No tool calls - returning final answer\n")
            return response.content
        
        # Execute all tool calls
        if verbose:
            print(f"Tool calls: {[tc['name'] for tc in response.tool_calls]}")
        
        for tool_call in response.tool_calls:
            tool_result = execute_tool(tool_call)
            messages.append(tool_result)
            if verbose:
                # Truncate long results for display
                content_preview = tool_result.content[:100] + "..." if len(tool_result.content) > 100 else tool_result.content
                print(f"  → {tool_call['name']}: {content_preview}")
        
        if verbose:
            print()
    
    return "Max iterations reached without final answer"


# =============================================================================
# MAIN: Demonstrate all methods
# =============================================================================

if __name__ == "__main__":
    # Test video URL
    test_url = "https://www.youtube.com/watch?v=T-D1OfcDW1M"
    
    print("=" * 70)
    print("YOUTUBE TOOLS AGENT DEMO")
    print("=" * 70)
    print()
    
    # Demonstrate each method (uncomment to run)
    
    # ----- Method 1: Manual -----
    # print(manual_summarize_video(test_url))
    
    # ----- Method 2: Chain -----
    # print(chain_summarize_video(test_url))
    
    # ----- Method 3: Agent Loop (Recommended) -----
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Summarize a specific video")
    print("=" * 70 + "\n")
    
    result = run_agent_loop(
        f"""Summarize this YouTube video: {test_url}

To do this:
1. First use extract_video_id to get the video ID
2. Then use fetch_transcript to get the transcript
3. Finally, summarize the content"""
    )
    print("RESULT:")
    print(result)
    
    # ----- More complex example -----
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Search and get metadata")
    print("=" * 70 + "\n")
    
    result2 = run_agent_loop("Search for 'Generative AI IBM' and get the metadata of the first result")
    print("RESULT:")
    print(result2)
