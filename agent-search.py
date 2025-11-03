import requests
import json

# IMPORTANT: Replace YOUR_API_KEY with your actual Gemini API key.
# This key is required to authenticate your request.
GEMINI_API_KEY = "YOUR_API_KEY"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
HEADERS = {'Content-Type': 'application/json'}
TIMEOUT = 10  # Timeout in seconds

def search_and_summarize(query: str, system_instruction: str = None):
    """
    Searches Google via the Gemini API with grounding enabled and returns
    the AI-generated summary and the web source citations.

    Args:
        query (str): The specific question or search query for the LLM.
        system_instruction (str, optional): An instruction to guide the
                                            model's persona or response format.
    """
    if GEMINI_API_KEY == "YOUR_API_KEY":
        print("ERROR: Please replace 'YOUR_API_KEY' with your actual Gemini API key.")
        return

    print(f"-> Querying Gemini API for: '{query}'")

    # 1. Define the core payload structure
    payload = {
        "contents": [{"parts": [{"text": query}]}],

        # 2. MANDATORY: Enable Google Search grounding tool
        "tools": [{
            "google_search": {}
        }],
    }

    # Add the system instruction if provided
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    # Add API key to the URL
    full_url = f"{API_URL}?key={GEMINI_API_KEY}"

    try:
        # Send the API request
        response = requests.post(full_url, headers=HEADERS, json=payload, timeout=TIMEOUT)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()

        candidate = result.get('candidates', [None])[0]

        if not candidate or not candidate.get('content') or not candidate['content'].get('parts'):
            print("API response was empty or malformed.")
            print(json.dumps(result, indent=2))
            return

        # 3. Extract the generated text summary
        summary_text = candidate['content']['parts'][0]['text']

        # 4. Extract grounding sources (citations)
        sources = []
        grounding_metadata = candidate.get('groundingMetadata')
        if grounding_metadata and grounding_metadata.get('groundingAttributions'):
            for attribution in grounding_metadata['groundingAttributions']:
                web = attribution.get('web')
                if web and web.get('uri') and web.get('title'):
                    sources.append({
                        "title": web['title'],
                        "uri": web['uri']
                    })

        print("\n--- AI Summary Response ---")
        print(summary_text)

        if sources:
            print("\n--- Grounding Sources (Citations) ---")
            for i, source in enumerate(sources, 1):
                print(f"{i}. {source['title']} ({source['uri']})")
        else:
            print("\n(No search sources were cited for this response.)")

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error occurred: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}")
    except Exception as e:
        print(f"An error occurred during response processing: {e}")


if __name__ == "__main__":
    # Example usage:
    search_topic = "What is the current price of gold per ounce?"
    analyst_prompt = "Act as a concise financial reporter. Summarize the answer in one direct sentence."
    search_and_summarize(search_topic, analyst_prompt)

    print("\n" + "="*50 + "\n")

    # Another example
    search_topic_2 = "Latest news on the James Webb Space Telescope discoveries."
    search_and_summarize(search_topic_2)
