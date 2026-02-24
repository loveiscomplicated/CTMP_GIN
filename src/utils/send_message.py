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
        print("오류: DISCORD_WEBHOOK_URL 환경 변수가 설정되지 않았습니다.")
        exit(1)

    # 2. 보낼 데이터 설정
    data = {
        "content": message,
        "username": "파이썬 봇"  # 선택 사항: 봇 이름 변경
    }

    # 3. POST 요청 보내기
    response = requests.post(webhook_url, json=data)

    # 4. 결과 확인
    if response.status_code == 204:
        print("메시지 전송 성공!")
    else:
        print(f"전송 실패: {response.status_code}")