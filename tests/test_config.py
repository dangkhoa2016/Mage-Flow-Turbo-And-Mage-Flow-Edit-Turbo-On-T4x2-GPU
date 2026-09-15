from __future__ import annotations

import math

import pytest

from server.config import (
    ConfigError,
    MAX_REQUEST_TIMEOUT_SECONDS,
    parse_tcp_port,
    parse_timeout_seconds,
    validate_loopback_http_url,
)
from server.workers.edit import EditWorkerConfig
from server.workers.t2i import T2IWorkerConfig

LOOPBACK_URLS = [
    "http://127.0.0.1:8101",
    "http://127.0.0.1:8101/",
    "http://localhost:8102",
    "http://localhost:8102/",
    "http://[::1]:8102",
]


def test_loopback_url_valid_and_normalized():
    assert validate_loopback_http_url("http://127.0.0.1:8101/", name="url") == "http://127.0.0.1:8101"
    assert validate_loopback_http_url("http://localhost:8102", name="url") == "http://localhost:8102"
    assert validate_loopback_http_url("http://[::1]:8102", name="url") == "http://[::1]:8102"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not a url",
        "https://127.0.0.1:8101",
        "http://0.0.0.0:8101",
        "http://example.com:8101",
        "http://192.168.1.10:8101",
        "http://10.0.0.5:8101",
        "http://172.16.0.5:8101",
        "http://user:pass@127.0.0.1:8101",
        "http://user@127.0.0.1:8101",
        "http://127.0.0.1:8101/api",
        "http://127.0.0.1:8101/?q=1",
        "http://127.0.0.1:8101/#frag",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:abc",
        "http://127.0.0.1",
        "file:///etc/passwd",
    ],
)
def test_loopback_url_rejects_invalid(raw):
    with pytest.raises(ConfigError):
        validate_loopback_http_url(raw, name="MAGE_FLOW_T2I_INTERNAL_URL")


def test_timeout_default_and_valid_override():
    assert parse_timeout_seconds(None, name="t") == 3600.0
    assert parse_timeout_seconds("60", name="t") == 60.0
    assert parse_timeout_seconds(7200, name="t") == 7200.0


@pytest.mark.parametrize(
    "bad",
    ["0", "-5", "nan", "inf", "-inf", "abc", "", "7200.1", str(MAX_REQUEST_TIMEOUT_SECONDS + 1)],
)
def test_timeout_rejects_invalid(bad):
    with pytest.raises(ConfigError):
        parse_timeout_seconds(bad, name="MAGE_FLOW_T2I_REQUEST_TIMEOUT_SECONDS")


def test_timeout_rejects_nonnumeric_type():
    with pytest.raises(ConfigError):
        parse_timeout_seconds(None, name="x", default="not-a-number")
    with pytest.raises(ConfigError):
        parse_timeout_seconds(math.nan, name="x")


def test_port_parsing_accepts_valid():
    assert parse_tcp_port("1", name="p") == 1
    assert parse_tcp_port("65535", name="p") == 65535
    assert parse_tcp_port(" 8101 ", name="p") == 8101


@pytest.mark.parametrize("bad", ["0", "-1", "65536", "abc", "", "  ", "1.5", "8101x"])
def test_port_parsing_rejects_invalid(bad):
    with pytest.raises(ConfigError):
        parse_tcp_port(bad, name="MAGE_FLOW_T2I_INTERNAL_PORT")


def test_worker_config_rejects_external_urls():
    with pytest.raises(ConfigError):
        T2IWorkerConfig.from_environment({"MAGE_FLOW_T2I_INTERNAL_URL": "http://0.0.0.0:8101"})
    with pytest.raises(ConfigError):
        EditWorkerConfig.from_environment({"MAGE_FLOW_EDIT_INTERNAL_URL": "https://example.com:8102"})
    with pytest.raises(ConfigError):
        EditWorkerConfig.from_environment({"MAGE_FLOW_EDIT_INTERNAL_URL": "http://user:pass@127.0.0.1:8102"})


def test_worker_config_rejects_bad_timeouts():
    with pytest.raises(ConfigError):
        T2IWorkerConfig.from_environment({"MAGE_FLOW_T2I_REQUEST_TIMEOUT_SECONDS": "0"})
    with pytest.raises(ConfigError):
        EditWorkerConfig.from_environment({"MAGE_FLOW_EDIT_REQUEST_TIMEOUT_SECONDS": "inf"})


def test_worker_config_accepts_loopback_ipv6_and_ports():
    cfg = T2IWorkerConfig.from_environment(
        {"MAGE_FLOW_T2I_INTERNAL_URL": "http://[::1]:8101", "MAGE_FLOW_T2I_REQUEST_TIMEOUT_SECONDS": "60"}
    )
    assert cfg.internal_url == "http://[::1]:8101"
    assert cfg.request_timeout_seconds == 60.0