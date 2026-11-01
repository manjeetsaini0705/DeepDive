from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()


# =========================
# Tavily Client
# =========================

tavily_api_key = os.getenv("TAVILY_API_KEY")

print("Tavily API key loaded:", bool(tavily_api_key))

tavily = TavilyClient(
    api_key=tavily_api_key
)


# =========================
# Search Tool
# =========================

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic.
    Returns titles, URLs and snippets.
    """

    print("\n>>> TAVILY SEARCH STARTED")
    print("Query:", query)

    try:
        results = tavily.search(
            query=query,
            max_results=5
        )

        print(">>> TAVILY SEARCH FINISHED")

        out = []

        for r in results["results"]:
            out.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['content'][:300]}\n"
            )

        return "\n----\n".join(out)

    except Exception as e:
        print(">>> TAVILY ERROR:", e)
        return f"Tavily search failed: {str(e)}"


# =========================
# Scraping Tool
# =========================

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL."""

    print("\n>>> SCRAPING URL:")
    print(url)

    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        soup = BeautifulSoup(
            resp.text,
            "html.parser"
        )

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        content = soup.get_text(
            separator=" ",
            strip=True
        )[:3000]

        print(">>> SCRAPING FINISHED")

        return content

    except Exception as e:
        print(">>> SCRAPING ERROR:", e)
        return f"Could not scrape URL: {str(e)}"