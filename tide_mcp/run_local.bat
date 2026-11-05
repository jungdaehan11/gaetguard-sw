@echo off
chcp 65001 >nul
cd /d %~dp0
rem 내 PC에서 조위 커넥터 켜기 (B 방법). 창 2개가 뜸: 터널 창 + 서버 창. 둘 다 켜 둔 동안만 커넥터가 동작함.
set HOST=127.0.0.1
if not exist mcp_secret.txt python -c "import secrets;open('mcp_secret.txt','w').write(secrets.token_urlsafe(24))"
python -m pip install -q -r requirements.txt
where cloudflared >nul 2>nul || (echo cloudflared 가 없습니다. 먼저: winget install --id Cloudflare.cloudflared & pause & exit /b)
start "tunnel - https://....trycloudflare.com 주소를 복사" cloudflared tunnel --url http://localhost:8000
python server.py
pause
