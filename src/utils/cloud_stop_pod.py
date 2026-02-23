import os
import sys
import requests


def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")

    if not api_key or not pod_id:
        print("ERROR: RUNPOD_API_KEY or RUNPOD_POD_ID not set")
        sys.exit(1)

    url = f"https://api.runpod.io/graphql?api_key={api_key}"

    payload = {
        "query": """
        mutation StopPod($podId: ID!) {
          stopPod(input: { podId: $podId })
        }
        """,
        "variables": {"podId": pod_id},
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
    except Exception as e:
        print("Request failed:", e)
        sys.exit(2)

    if response.status_code != 200:
        print("HTTP error:", response.status_code, response.text)
        sys.exit(3)

    data = response.json()

    if "errors" in data:
        print("GraphQL error:", data["errors"])
        sys.exit(4)

    print("Pod stop requested successfully.")


if __name__ == "__main__":
    main()