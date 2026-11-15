# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

prefcheck 는 네트워크 서버를 열지 않습니다. `--tokenizer` 를 쓸 때만 Hugging Face Hub 에 접속하며(`--offline` 으로 차단 가능), `trust_remote_code` 는 쓰지 않습니다. / prefcheck does not open a network server. It contacts the Hugging Face Hub only when you pass `--tokenizer` (block it with `--offline`), and it never uses `trust_remote_code`.
