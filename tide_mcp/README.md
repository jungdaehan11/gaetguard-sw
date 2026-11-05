# 갯가드 조위 커넥터 (MCP 서버)

Claude가 국립해양조사원 조위 API를 **직접** 부르게 해 주는 작은 서버입니다. law2026처럼 claude.ai의 사용자 지정 커넥터로 붙입니다.
인증키는 서버 환경변수에만 두고, 저장소에는 올리지 않습니다.

## 먼저 알아 둘 약점

1. 해외 서버(Render Singapore)에서 공공데이터포털 접속은 **2026-11-05에 됨을 확인**했습니다. 나중에 막히면 도구가 "연결하지 못했습니다"나 HTTP 403 오류를 냅니다 → 그때는 B(내 PC) 방법을 쓰세요.
2. Render 무료 서버는 15분 동안 요청이 없으면 멈추고, 다시 켜지는 데 약 1분 걸립니다 ([Render 무료 안내](https://render.com/docs/free)). 첫 호출이 실패하면 1분 뒤 다시 하거나, 쓰기 전에 서버 주소를 브라우저로 한 번 열어 두세요 (`ok`가 나오면 깨어난 것).
3. 서버를 막는 장치는 **주소 속 비밀 문자열뿐**입니다. 커넥터 주소를 남에게 주면 그 사람이 내 인증키 호출 한도를 씁니다.
4. Claude 무료 요금제는 사용자 지정 커넥터를 1개만 쓸 수 있습니다 ([Claude 도움말](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)). law2026이 이미 있으면 추가가 안 될 수 있습니다.
5. 관측소 코드 `DT_0043` → 영흥도, `DT_0052` → 인천송도, `DT_0001` → 인천은 실제 API가 돌려준 이름으로 확인했습니다 (2026-11-05). 다른 관측소 코드는 아직 모릅니다.
6. `station_tide`는 "조위관측소 실측·예측 조위"(공공데이터포털 15142507) 활용신청을 따로 해야 합니다.
7. 2026-11-05 Render 배포 후 claude.ai 커넥터로 도구 3개를 실제로 불러 확인했습니다. 실측·예측 API에서 미래 시각은 `bscTdlvHgt`가 비어 있고 값이 정수 cm라서, `bscTdlvHgt` = 실측이 맞는 것으로 보입니다.

## 도구 3개

| 도구 | 하는 일 | 쓰는 API |
|---|---|---|
| `tide_point_summary(lat, lon, date)` | 위경도 한 점의 하루 고조·저조 시각과 높이, 가장 가까운 기준항, 조시차·조고비·평균해면, 기준항과의 시각 차이(분) | TideBED (1분 간격 예측) |
| `tide_point_series(lat, lon, date, interval_min, start, end, include_reference)` | 위경도 한 점의 예측 조위 시계열 (최대 300줄) | TideBED |
| `station_tide(obs_code, date, interval_min, start, end)` | 관측소 실측·예측 조위, 차이(실측-예측) 평균·최대·최소, 마지막 실측값 | 조위관측소 실측·예측 조위 |

- 날짜는 `20261106` 또는 `2026-11-06`, 비우면 오늘(한국 시각)
- 실측·예측 API의 필드 뜻은 공공데이터포털 명세 기준입니다: `tdlvHgt` = 예측조위, `bscTdlvHgt` = 실측조위 (이름과 반대처럼 보여서 적어 둠)
- 같은 요청은 서버가 기억해 둡니다 (예측 6시간, 실측 2분) → 호출 한도를 덜 씀

## 준비물 두 가지

1. 공공데이터포털 인증키 (일반 인증키. Encoding/Decoding 어느 쪽이든 됨)
2. 비밀 문자열 (아무도 못 맞힐 긴 글자). cmd에서 만들기:
   ```
   python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```

## A. Render에 올리기 (권장 — 항상 같은 주소)

1. https://render.com 에 GitHub 계정으로 가입
2. **New → Web Service** → 저장소 `jungdaehan11/gaetguard-sw` 선택 (공개 저장소라 주소만 넣어도 됨)
3. 설정 (화면 이름은 조금 다를 수 있음)
   | 항목 | 값 |
   |---|---|
   | Name | `gaetguard-tide` (주소가 됨) |
   | Region | Singapore (가장 가까운 곳) |
   | Branch | `main` |
   | Root Directory | `tide_mcp` |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `python server.py` |
   | Instance Type | Free |
4. **Environment Variables**에 두 개 추가
   - `DATA_GO_KR_KEY` = 공공데이터포털 인증키
   - `MCP_SECRET` = 위에서 만든 비밀 문자열
5. Deploy → 끝나면 위쪽에 `https://gaetguard-tide.onrender.com` 같은 주소가 나옴. 브라우저로 열어 `ok`가 보이면 성공
6. 커넥터 주소 = `https://<그 주소>/<비밀 문자열>/mcp`

## B. 내 PC에서 켜기 (A에서 공공데이터포털이 막힐 때)

내 PC는 한국 인터넷이라 공공데이터포털이 열립니다. 대신 **PC와 창 2개를 켜 둔 동안만** 동작하고, 켤 때마다 주소가 바뀝니다.

1. cloudflared 설치 (한 번만): cmd에서 `winget install --id Cloudflare.cloudflared`
2. `gaetguard` 폴더에 `datago_key.txt` (인증키 한 줄)가 있는지 확인
3. `gaetguard\tide_mcp\run_local.bat` 더블클릭
   - 터널 창에 `https://xxxx.trycloudflare.com` 주소가 나옴
   - 서버 창에 `커넥터 주소 = <터널 주소>/<비밀>/mcp` 가 나옴
4. 커넥터 주소 = 터널 주소 + `/<비밀>/mcp`

## claude.ai에 커넥터 추가 (law2026 넣을 때와 같음)

1. **사용자 지정(Customize) → 커넥터 → + 추가 → 사용자 지정 커넥터 추가**
2. 이름 `갯가드 조위`, URL = 위의 커넥터 주소 → 계속
3. 인증은 **로그인 없음(No sign in)** → 추가
4. 새 대화에서 커넥터를 켜고 물어보기

시험 질문 예:
- "갯가드 조위로 위도 37.340, 경도 126.675의 2026-11-06 고조·저조 알려줘"
- "같은 지점 06:00~09:00 10분 간격 조위 표로"
- "인천 관측소 오늘 실측과 예측 차이"

## 파일

| 파일 | 내용 |
|---|---|
| `server.py` | 서버 (Python, MCP SDK 1.x `FastMCP`, Streamable HTTP) |
| `requirements.txt` | `mcp>=1.25,<2` (2.x는 이름이 바뀌어 안 돌아감) |
| `run_local.bat` | B 방법: 내 PC에서 서버 + 터널 켜기 |

출처 표시: 국립해양조사원, 공공데이터포털 OpenAPI. 실측·예측 조위 API는 "공공저작물 출처표시(제1유형)"로 공공데이터포털에 적혀 있음. TideBED의 이용 조건은 [확인 필요].
