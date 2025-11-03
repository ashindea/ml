import requests
import json
from time import sleep

# IMPORTANT: Replace YOUR_API_KEY with your actual Gemini API key.
# This key is required to authenticate your request.
GEMINI_API_KEY = "YOUR_API_KEY"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
HEADERS = {'Content-Type': 'application/json'}
TIMEOUT = 30  # Increased timeout for complex searches/planning

def call_gemini_api(payload, full_url, retries=3):
    """Handles the API call with basic retry logic."""
    for attempt in range(retries):
        try:
            response = requests.post(full_url, headers=HEADERS, json=payload, timeout=TIMEOUT)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error occurred on attempt {attempt + 1}: {errh}")
        except requests.exceptions.RequestException as err:
            print(f"Request Error occurred on attempt {attempt + 1}: {err}")

        if attempt < retries - 1:
            # Exponential backoff: sleep 2, 4 seconds before retrying
            sleep(2 ** (attempt + 1))
        else:
            return None # Fail after last retry

    return None


def plan_trip(city: str, days: int, departure_city: str = "New York (JFK/LGA/EWR)"):
    """
    Creates a travel plan including a flight summary and a daily itinerary
    for a given city and duration, using Google Search grounding.

    Args:
        city (str): The destination city.
        days (int): The duration of the trip in days.
        departure_city (str, optional): The major hub to search flights from.
    """
    if GEMINI_API_KEY == "YOUR_API_KEY":
        print("ERROR: Please replace 'YOUR_API_KEY' with your actual Gemini API key.")
        return

    print(f"\n--- Planning a {days}-day trip to {city} from {departure_city} ---")

    # 1. Define the system instruction to set the agent's persona and structure
    system_instruction = (
        "You are a professional AI travel agent and itinerary planner. "
        "Your response must be highly structured and contain only two main Markdown headings: "
        "'## Flight Summary' and '## Daily Itinerary'."
    )

    # 2. Define the user query, explicitly asking for search information
    user_query = (
        f"I am planning a trip to {city} for {days} days. "
        "Please use your search tool to find the following and compile it into a single response:\n"
        f"1. Key attractions and itinerary suggestions for a {days}-day trip to {city}.\n"
        f"2. Available flight options (including typical price range, major airlines, and travel time) "
        f"from {departure_city} to {city} for travel next month (assume a 7-day window).\n\n"
        "Create the final travel plan based on the search results, ensuring the itinerary is detailed and relevant."
    )

    # 3. Construct the payload
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},

        # MANDATORY: Enable Google Search grounding tool
        "tools": [{"google_search": {}}],
    }

    # 4. Perform the API call
    full_url = f"{API_URL}?key={GEMINI_API_KEY}"
    result = call_gemini_api(payload, full_url)

    if not result:
        print("Failed to get a response from the Gemini API after multiple retries.")
        return

    # 5. Process the response
    try:
        candidate = result.get('candidates', [None])[0]

        if not candidate or not candidate.get('content') or not candidate['content'].get('parts'):
            print("API response was empty or malformed.")
            print(json.dumps(result, indent=2))
            return

        # Extract the generated travel plan text
        plan_text = candidate['content']['parts'][0]['text']

        # Extract grounding sources (citations)
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

        print("\n" + "="*70)
        print("          AI GENERATED TRAVEL PLAN")
        print("="*70)
        print(plan_text)
        print("="*70)

        if sources:
            print("\n--- Grounding Sources (Citations) ---")
            for i, source in enumerate(sources, 1):
                print(f"{i}. {source['title']} ({source['uri']})")
        else:
            print("\n(No specific search sources were cited for this response.)")

    except Exception as e:
        print(f"An error occurred during response processing: {e}")


if __name__ == "__main__":
    # --- Example 1: A 5-day international trip ---
    plan_trip(city="Tokyo, Japan", days=5, departure_city="Los Angeles")

    print("\n" + "#"*70 + "\n")

    # --- Example 2: A 3-day domestic trip ---
    plan_trip(city="Seattle, WA", days=3, departure_city="Dallas")
