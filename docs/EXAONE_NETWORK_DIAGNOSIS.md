# EXAONE 다운로드 DNS 오류 진단 — NET-EXAONE-01

## 확인 결과

Hugging Face 전용 장애나 프록시 오설정으로 확인되지 않았다. 이 실행 프로세스의 외부 네트워크 접근이 거부된다.

- localhost만 정상 해석.
- huggingface.co / pypi.org / www.microsoft.com 모두 getaddrinfo11001.
- DNS를 사용하지 않는1.1.1.1:443 /8.8.8.8:443 직접 접속도 WinError10013 접근 거부.
- 자체 토큰의 TokenIsAppContainer=True.
- Resolve-DnsName도 액세스 거부. 네트워크 설정 CIM 조회도 권한 거부.
- 사용자 config.toml의 sandbox_workspace_write.network_access는 이미 true.
- HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/HF_ENDPOINT 및 offline 환경 변수는 설정돼 있지 않음.

따라서 현재 확인된 차단은 실제 실행 세션의 sandbox/network permission 영역이다. PC DNS 서버 변경, hosts 고정, SSL 검증 해제, 방화벽 비활성화로 우회하지 않는다. 이 세션의 권한을 코드에서 자체 승격하지 않았다.

이는 호스트의 일반 PowerShell 연결을 검사한 결과가 아니다. 호스트 DNS가 정상인지 여부는 별도로 확인해야 한다. 설정 파일의 true만으로 현재 이미 시작된 세션의 실효 네트워크 접근을 보장하지 않는다.

## 재현

프로젝트 루트 PowerShell:

    .\.venv-server\Scripts\python.exe -B -X utf8 tools/check_network.py

종료0: Hugging Face 모델 API의 DNS/HTTPS 성공.
종료2: AppContainer에서 직접 IP 접속까지 접근 거부.
종료1: 그 밖의 DNS/HTTP 장애. JSON의 실제 원인을 확인한다.

이 도구는 OS DNS/hosts/firewall/proxy 설정을 변경하지 않는다. 저장한 진단을 덮어쓰지 않는다.

## 현재 가능한 재개 방법

현재 세션은 도구의 네트워크 권한이 restricted이고 sandbox 승인 확대도 허용되지 않는다. 도구나 셸 자식 프로세스를 통해 이 제한을 스스로 해제할 수 없다.

사용자가 직접 연 일반 Windows PowerShell에서 다음을 실행하면 이 세션 밖에서 연결 확인 후, 이미 구현한 공식 EXAONE 다운로드를 시도할 수 있다. 관리자 권한이나 전역 DNS 변경은 필요하지 않다. 성공 여부는 실제 실행 후에만 확정한다.

    Set-Location C:\Projects\ML_midterm
    .\.venv-server\Scripts\python.exe -B -X utf8 tools/check_network.py
    if ($LASTEXITCODE -eq 0) {
        .\.venv-server\Scripts\python.exe -B -X utf8 tools/fetch_exaone.py --experiment-id EXAONE-BASE-FETCH-02
    }

다운로드 위치는 artifacts/pretrained/exaone-4.0-1.2b. 요청 모델 이외의 가중치로 대체하지 않는다. 원본 데이터 접근·학습·ZIP 생성은 이 명령에 없다. 네트워크 허용 세션으로 다시 시작하는 방법은 앱/관리 정책에서 제어해야 하며 현재 세션에서 변경한 것으로 표시하지 않는다.

## 공식 문서와 근거

- [Windows sandbox](https://learn.chatgpt.com/docs/windows/windows-sandbox): 실제 permission profile이 파일/네트워크 접근을 결정.
- [Agent approvals and security](https://learn.chatgpt.com/docs/agent-approvals-security): workspace-write의 network_access와 실행 네트워크 권한.
- [Permission profiles](https://learn.chatgpt.com/docs/permissions): 새 profile 방식과 legacy sandbox 설정의 구분.
- [실측 진단](../reports/network/NET-EXAONE-01/diagnosis.json).

현재 상태는 원인 진단 완료, 세션 네트워크 권한으로 연결 복구 차단이다. 실제 모델 다운로드/학습은 아직 재개되지 않았다.
