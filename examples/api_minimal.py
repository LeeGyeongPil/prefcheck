"""Minimal library example: check rows in memory without any tokenizer (approximate mode)."""

from prefcheck import CheckConfig, check_rows

rows = [
    {
        "prompt": [{"role": "user", "content": "What color is the sky?"}],
        "chosen": [{"role": "assistant", "content": "The sky is blue."}],
        "rejected": [{"role": "assistant", "content": "The sky is blue."}],  # identical pair: an error
    },
    {
        "prompt": [{"role": "user", "content": "Name a prime number."}],
        "chosen": [{"role": "assistant", "content": "Seven is a prime number."}],
        "rejected": [{"role": "assistant", "content": "Nine is a prime number."}],
    },
]

report = check_rows(rows, CheckConfig(max_length=64))
print(f"has_errors={report.has_errors} findings={len(report.findings)}")
for finding in report.findings:
    print(f"row {finding.row}: {finding.rule} ({finding.severity})")
