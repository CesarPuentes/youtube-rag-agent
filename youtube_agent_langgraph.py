#!/usr/bin/env python3
"""
YouTube Tools Agent (LangGraph Version)
=======================================
This script uses LangGraph's `create_react_agent` to automate the Reasoning + Acting loop.
"""

import re
import warnings
import logging
from typing import List, Dict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Search
import yt_dlp
from langgraph.prebuilt import create_react_agent

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger('pytube').setLevel(logging.ERROR)
logging.getLogger('yt_dlp').setLevel(logging.ERROR)
yt_dlp_logger = logging.getLogger('yt_dlp')

# =============================================================================
# TOOLS
# =============================================================================

@tool
def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from a URL."""
    match = re.search(r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else "Error: Invalid YouTube URL"

@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """Fetches the transcript of a YouTube video given its ID."""
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        return " ".join([s.text for s in transcript.snippets])
    except Exception as e:
        return f"Error: {e}"

@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """Search YouTube for videos matching the query."""
    try:
        return [{"title": yt.title, "video_id": yt.video_id, "url": f"https://youtu.be/{yt.video_id}"} 
                for yt in Search(query).results]
    except Exception as e:
        return f"Error: {e}"

@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata from a YouTube URL (title, views, duration, channel, likes, comments, chapters)."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dlp_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {k: info.get(v) for k, v in [
            ('title', 'title'), ('views', 'view_count'), ('duration', 'duration'),
            ('channel', 'uploader'), ('likes', 'like_count'), ('comments', 'comment_count'),
            ('chapters', 'chapters')
        ]}

@tool
def get_thumbnails(url: str) -> List[Dict]:
    """Get available thumbnails for a YouTube video."""
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dlp_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            return [{"url": t['url'], "resolution": f"{t.get('width', '')}x{t.get('height', '')}"} 
                    for t in info.get('thumbnails', []) if 'url' in t]
    except Exception as e:
        return [{"error": str(e)}]

# =============================================================================
# AGENT SETUP
# =============================================================================

tools = [extract_video_id, fetch_transcript, search_youtube, get_full_metadata, get_thumbnails]
llm = ChatOllama(model="qwen2.5:7b", temperature=0.5) # Tools are bound inside create_react_agent

# =============================================================================
# LANGGRAPH REACT AGENT
# =============================================================================

# This single line replaces the entire manual 'for' loop and message management.
# It automatically handles tool-calling logic and tool-message pairing.
langgraph_agent = create_react_agent(llm, tools=tools)

def agent(query: str, verbose: bool = True) -> str:
    """
    Agent implementation using LangGraph's create_react_agent.
    """
    if verbose:
        print(f"--- Running Query with LangGraph ---\nQuery: {query}\n")
        
    # LangGraph returns a dictionary containing all state variables (like messages)
    result = langgraph_agent.invoke({"messages": [("human", query)]})
    
    # We return the last message content (the agent's final answer)
    return result["messages"][-1].content

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Summarizing video with LangGraph Agent...")
    print("=" * 60)
    
    query = "Summarize this YouTube video: https://www.youtube.com/watch?v=T-D1OfcDW1M"
    result = agent(query)
    
    print("\nFINAL ANSWER:\n" + result)
