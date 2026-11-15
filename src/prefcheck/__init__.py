"""prefcheck: lint DPO/ORPO preference datasets with your trainer's tokenizer and truncation settings.

Quick start::

    from prefcheck import CheckConfig, check_rows

    report = check_rows(rows, CheckConfig(max_length=1024))
    print(report.has_errors, len(report.findings))
"""

from ._version import __version__
from .checker import check_file, check_rows
from .config import CheckConfig, Thresholds
from .errors import ConfigError, InputFormatError, PrefcheckError, TokenizerLoadError
from .report import Finding, Report
from .rules import RULES
from .tokenization import ApproxTokenizer, HFTokenizer, Tokenizer, load_tokenizer

__all__ = [
    "__version__",
    "check_rows",
    "check_file",
    "CheckConfig",
    "Thresholds",
    "load_tokenizer",
    "ApproxTokenizer",
    "HFTokenizer",
    "Tokenizer",
    "Report",
    "Finding",
    "RULES",
    "PrefcheckError",
    "InputFormatError",
    "TokenizerLoadError",
    "ConfigError",
]
