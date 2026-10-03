import os

from app.config import CameraConfig, OcrConfig


def test_builds_rtsp_url_from_parts(monkeypatch):
    monkeypatch.setenv("CAMERA_IP", "10.0.0.5")
    monkeypatch.setenv("CAMERA_RTSP_PORT", "554")
    monkeypatch.setenv("CAMERA_USERNAME", "admin")
    monkeypatch.setenv("CAMERA_PASSWORD", "secret")
    monkeypatch.setenv("CAMERA_RTSP_PATH", "/Streaming/Channels/102")
    monkeypatch.delenv("CAMERA_RTSP_URL", raising=False)

    cfg = CameraConfig()
    assert cfg.rtsp_url == "rtsp://admin:secret@10.0.0.5:554/Streaming/Channels/102"
    assert "secret" not in cfg.rtsp_url_masked
    assert "***" in cfg.rtsp_url_masked


def test_explicit_url_overrides_parts(monkeypatch):
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://foo:bar@1.2.3.4:554/custom")
    monkeypatch.setenv("CAMERA_IP", "10.0.0.5")

    cfg = CameraConfig()
    assert cfg.rtsp_url == "rtsp://foo:bar@1.2.3.4:554/custom"


def test_password_with_at_symbol_is_url_encoded(monkeypatch):
    # A literal '@' in the password would otherwise be misread as the
    # userinfo/host separator by the RTSP URL parser.
    monkeypatch.setenv("CAMERA_IP", "172.29.7.7")
    monkeypatch.setenv("CAMERA_RTSP_PORT", "554")
    monkeypatch.setenv("CAMERA_USERNAME", "admin")
    monkeypatch.setenv("CAMERA_PASSWORD", "Kanishk@2216")
    monkeypatch.setenv("CAMERA_RTSP_PATH", "/Streaming/Channels/102")
    monkeypatch.delenv("CAMERA_RTSP_URL", raising=False)

    cfg = CameraConfig()
    url = cfg.rtsp_url
    # Exactly one '@' should separate userinfo from host.
    assert url.count("@") == 1
    assert url.endswith("@172.29.7.7:554/Streaming/Channels/102")
    assert "Kanishk%402216" in url
    assert "Kanishk@2216" not in cfg.rtsp_url_masked
    assert "***" in cfg.rtsp_url_masked


def test_missing_ip_and_url_raises(monkeypatch):
    monkeypatch.delenv("CAMERA_RTSP_URL", raising=False)
    monkeypatch.delenv("CAMERA_IP", raising=False)

    cfg = CameraConfig()
    try:
        _ = cfg.rtsp_url
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ocr_mkldnn_unset_defers_to_platform_default(monkeypatch):
    # None (not False) is what tells OCRDetector to choose per-platform --
    # off on Windows, where paddle's oneDNN kernels crash, on elsewhere.
    monkeypatch.delenv("OCR_ENABLE_MKLDNN", raising=False)
    assert OcrConfig().enable_mkldnn is None


def test_ocr_mkldnn_explicit_override_is_respected(monkeypatch):
    monkeypatch.setenv("OCR_ENABLE_MKLDNN", "false")
    assert OcrConfig().enable_mkldnn is False

    monkeypatch.setenv("OCR_ENABLE_MKLDNN", "true")
    assert OcrConfig().enable_mkldnn is True
