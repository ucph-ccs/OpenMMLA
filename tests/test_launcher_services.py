from openmmla.tui.screens import launcher


def _service(name):
    return next(service for service in launcher._build_service_registry("/tmp/project") if service.name == name)


def test_asr_server_uses_nemo_environment_name():
    service = _service("ASR Server")

    assert service.conda_env == "asr-server-nemo"


def test_mllm_server_is_registered_as_vfa_runtime_service():
    service = _service("MLLM Server")

    assert service.category == "VFA"
    assert service.conda_env == "vfa-vllm"
    assert service.config_dir == "/tmp/project"
    assert service.launch_type == "vllm"
    assert "OpenAI-compatible vLLM server" in service.description


def test_mllm_server_does_not_require_pipeline_config_file():
    assert launcher._service_requires_config(_service("VFA Server"))
    assert not launcher._service_requires_config(_service("MLLM Server"))


def test_mllm_server_command_uses_qwen3_vl_on_8010():
    command = launcher._vllm_serve_command()

    assert "Qwen/Qwen3-VL-8B-Instruct" in command
    assert "--port 8010" in command
    assert "--limit-mm-per-prompt" in command
    assert "'{\"image\":4}'" in command
    assert "--api-key EMPTY" in command


def test_mllm_server_command_accepts_config_overrides():
    command = launcher._vllm_serve_command({
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "host": "127.0.0.1",
        "port": 9010,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "limit_mm_per_prompt": '{"image":2}',
        "gpu_memory_utilization": 0.7,
        "api_key": "test-key",
    })

    assert "Qwen/Qwen3-VL-4B-Instruct" in command
    assert "--host 127.0.0.1" in command
    assert "--port 9010" in command
    assert "--max-model-len 4096" in command
    assert "'{\"image\":2}'" in command
    assert "--gpu-memory-utilization 0.7" in command
    assert "--api-key test-key" in command


def test_mllm_config_loads_saved_model_and_port(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "mllm_server.yml").write_text(
        "server:\n"
        "  model: Qwen/Qwen3-VL-4B-Instruct\n"
        "  port: 9010\n",
        encoding="utf-8",
    )

    config = launcher._mllm_config(str(tmp_path))

    assert config["model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert config["port"] == 9010
    assert config["host"] == "0.0.0.0"
