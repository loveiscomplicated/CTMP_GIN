import os
import sys
import time

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False

def _retry_after_seconds(response: requests.Response, attempt: int) -> float | None:
    if response.status_code == 429:
        for header in ("Retry-After", "X-RateLimit-Reset-After"):
            value = response.headers.get(header)
            if value is not None:
                try:
                    return max(float(value), 0.0) + 0.1
                except ValueError:
                    pass
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        retry_after = payload.get("retry_after")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0) + 0.1
            except (TypeError, ValueError):
                pass
        return 1.0
    if 500 <= response.status_code < 600:
        return min(2.0 ** attempt, 10.0)
    return None


def post_discord_message(
    webhook_url: str,
    message: str,
    bot_name: str = "Python Bot",
    *,
    max_attempts: int = 5,
    timeout: int = 15,
) -> bool:
    data = {
        "content": message,
        "username": bot_name,
    }

    last_error = ""
    for attempt in range(max_attempts):
        try:
            response = requests.post(webhook_url, json=data, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"request error: {exc}"
            retry_delay = min(2.0 ** attempt, 10.0)
        else:
            if response.status_code == 204:
                return True
            last_error = f"status={response.status_code} body={response.text}"
            retry_delay = _retry_after_seconds(response, attempt)

        if retry_delay is None or attempt >= max_attempts - 1:
            break
        print(
            f"Discord notification failed ({last_error}); retrying in {retry_delay:.2f}s",
            file=sys.stderr,
        )
        time.sleep(retry_delay)

    print(
        f"Discord notification failed after {max_attempts} attempt(s): {last_error}. "
        "Continuing without Discord notification.",
        file=sys.stderr,
    )
    return False


def send_discord_message(message: str, bot_name: str = "Python Bot") -> bool:
    # 현재 실행 중인 파이썬 파일의 부모 디렉토리를 찾습니다.
    cur_dir = os.path.dirname(__file__)
    env_path = os.path.join(cur_dir, '..', '..', '.env')

    # 해당 경로의 파일을 명시적으로 로드합니다.
    load_dotenv(dotenv_path=env_path, override=True)

    # 환경 변수에서 웹훅 URL 가져오기
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    # 웹훅 URL이 설정되었는지 확인
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL environment variable is not set.")
        return False

    if post_discord_message(webhook_url, message, bot_name):
        print("Message Send succeed!")
        return True
    return False

if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello from RunPod"
    send_discord_message(msg, bot_name="RunPod Bot")
