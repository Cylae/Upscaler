from src.core.detector import get_hardware_info


def test_hardware_detector():
    hw = get_hardware_info()
    assert hw.ffmpeg_path is not None
    assert hw.has_nvidia is True
    assert "GeForce" in hw.gpu_name or "RTX" in hw.gpu_name
    assert hw.has_nvenc_hevc is True
    assert hw.has_nvenc_h264 is True
    assert hw.realesrgan_bin is not None
    assert hw.realcugan_bin is not None
