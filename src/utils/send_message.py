import requests
import os
from dotenv import load_dotenv

def send_dischord_message(message: str):
    # .env 파일에서 환경 변수 로드
    load_dotenv()

    # 환경 변수에서 웹훅 URL 가져오기
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    # 웹훅 URL이 설정되었는지 확인
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL environment variable is not set.")
        exit(1)

    # 2. 보낼 데이터 설정
    data = {
        "content": message,
        "username": "PYTHON BOT"  # name of bot
    }

    # 3. send POST request
    response = requests.post(webhook_url, json=data)

    # 4. 결과 확인
    if response.status_code == 204:
        print("Message Send succeed!")
    else:
        print(f"FAILED Sending: {response.status_code}")