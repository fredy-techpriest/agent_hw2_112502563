import os
from typing import Any, Dict, List

import requests


def tavily_query(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Query Tavily search API and return normalized results.

    Required environment variable:
    - TAVILY_API_KEY
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query.strip(),
            "max_results": max_results,
        },
        timeout=30,
    )
    response.raise_for_status()# 檢查回應狀態

    data = response.json()
    results: List[Dict[str, Any]] = data.get("results", [])

    return {
        "query": query.strip(),
        "answer": data.get("answer", ""),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in results
        ],
    }
