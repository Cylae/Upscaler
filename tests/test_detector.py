from unittest.mock import patch
from src.core.detector import get_hardware_info, HardwareInfo, clear_hardware_cache


def test_hardware_detector():
    hw = get_hardware_info()
    assert isinstance(hw, HardwareInfo)
    assert hw.cpu_cores >= 1
    assert isinstance(hw.gpu_name, str)
    assert isinstance(hw.has_nvidia, bool)

    # Clear cache and test with mocked hardware environment to verify detector contract
    clear_hardware_cache()
    with patch("src.core.hardware.detect_nvidia_gpu", return_value=("NVIDIA GeForce RTX 3070", 8.0, True)), \
         patch("src.core.hardware.detect_ffmpeg", return_value=("/usr/bin/ffmpeg", "/usr/bin/ffprobe", "NVIDIA NVENC FFmpeg 6.0", True, True)), \
         patch("src.core.hardware.detect_ai_binaries", return_value=("tools/realesrgan/realesrgan-ncnn-vulkan", "tools/realcugan/realcugan-ncnn-vulkan")):
        mocked_hw = get_hardware_info(force_refresh=True)
        assert mocked_hw.ffmpeg_path is not None
        assert mocked_hw.has_nvidia is True
        assert "GeForce" in mocked_hw.gpu_name or "RTX" in mocked_hw.gpu_name
        assert mocked_hw.has_nvenc_hevc is True
        assert mocked_hw.has_nvenc_h264 is True
        assert mocked_hw.realesrgan_bin is not None
        assert mocked_hw.realcugan_bin is not None

    clear_hardware_cache()
