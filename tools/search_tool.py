from langchain_classic.utilities import SerpAPIWrapper
from config.settings import SERPAPI_API_KEY

if SERPAPI_API_KEY is None:
    print("Warning: SERPAPI_API_KEY not set. Web search will fail until configured.")

serp = SerpAPIWrapper()

def web_search_snippets(query: str, num_results: int = 5) -> list:
    try:
        raw = serp.run(query)
        return [raw]
    except Exception as e:
        return [f"Search error: {str(e)}"]
