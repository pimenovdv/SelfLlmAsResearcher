import urllib.request
import json

def resolve_library_id(query):
    """
    Resolves a general query to a specific library ID (PyPI package name).
    """
    query = query.lower()
    if 'transformer' in query and 'lens' in query:
        return 'transformer-lens'
    elif 'torch' in query or 'pytorch' in query:
        return 'torch'
    elif 'einops' in query:
        return 'einops'
    elif 'transformers' in query or 'huggingface' in query:
        return 'transformers'
    elif 'accelerate' in query:
        return 'accelerate'
    elif 'openai' in query:
        return 'openai'
    else:
        # If it doesn't match known mappings, return the query as is (assume it's a PyPI package name)
        # Replacing spaces with hyphens just in case
        return query.strip().replace(" ", "-")

def get_library_docs(library_id, topic=None):
    """
    Returns up-to-date documentation for a given library ID from PyPI.
    If topic is provided, acts as a lightweight RAG by returning relevant sections.
    Limits the response size to avoid context overflow.
    """
    url = f"https://pypi.org/pypi/{library_id}/json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                info = data.get('info', {})
                desc = info.get('description', '')

                if not desc or desc == 'UNKNOWN':
                    desc = info.get('summary', 'No detailed description available.')

                if topic:
                    topic_lower = topic.lower()
                    paragraphs = desc.split('\n\n')
                    relevant_paragraphs = [p for p in paragraphs if topic_lower in p.lower()]
                    if relevant_paragraphs:
                        docs = f"Documentation for {library_id} on topic '{topic}':\n\n" + "\n\n".join(relevant_paragraphs)
                    else:
                        docs = f"Topic '{topic}' not found in the main documentation. Here is the general description:\n\n{desc}"
                else:
                    docs = f"Documentation for {library_id}:\n\n{desc}"

                # Truncate to prevent context overflow (e.g. 4000 characters)
                if len(docs) > 4000:
                    docs = docs[:4000] + "\n...[truncated due to length]"
                return docs
            else:
                return f"Failed to fetch documentation for '{library_id}'. HTTP status: {response.status}"
    except Exception as e:
         return f"Error fetching documentation for '{library_id}': {str(e)}"
