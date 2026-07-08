# Dockerized ASR / VFA services

One image per service, so每个服务的 Python 环境完全隔离，升级互不影响。
端口与 nginx 网关的 upstream 完全一致，网关配置无需改动。

| Service | Image | Port | GPU | Stack |
|---|---|---|---|---|
| AudioInferer (wespeaker) | `openmmla/asr-audio-inferer-wespeaker` | 5001 | ✅ | torch 2.4.1 + wespeaker |
| AudioInferer (nemo, 可选) | `openmmla/asr-audio-inferer-nemo` | 5001 | ✅ | nemo-toolkit ≤1.23 |
| AudioResampler | `openmmla/asr-audio-resampler` | 5002 | — | librosa (CPU) |
| SpeechEnhancer | `openmmla/asr-speech-enhancer` | 5003 | ✅ | torch 2.4.1 + denoiser |
| SpeechSeparator | `openmmla/asr-speech-separator` | 5004 | ✅ | torch 2.4.1 + modelscope |
| SpeechTranscriber | `openmmla/asr-speech-transcriber` | 5005 | ✅ | **whisperx 3.8.6 + torch 2.8 + ct2≥4.5 (cuDNN 9)** |
| VoiceActivityDetector | `openmmla/asr-voice-activity-detector` | 5006 | — | silero-vad (CPU torch) |
| VLLMFrameAnalyzer | `openmmla/vfa-frame-analyzer` | 5007 | ✅ | torch 2.7 + tf-keras/retina-face |
| vLLM VLM 后端 (可选) | `vllm/vllm-openai` | 8000 | ✅ | 官方镜像, profile `mllm` |

## 宿主机要求 (server-01)

- NVIDIA 驱动（已有 595.71.05）
- Docker Engine + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `~/.openmmla/master.key`（解密 config 里的 ENC(...) 值，只读挂载进容器）

## 使用

在仓库根目录执行：

```bash
# ASR 全部服务（构建 + 启动）
docker compose -f docker/docker-compose.asr.yml up -d --build

# VFA frame analyzer
docker compose -f docker/docker-compose.vfa.yml up -d --build

# 查看状态 / 日志
docker compose -f docker/docker-compose.asr.yml ps
docker compose -f docker/docker-compose.asr.yml logs -f speech-transcriber

# 停止
docker compose -f docker/docker-compose.asr.yml down
```

### 切换 inferer 后端（wespeaker ⇄ nemo）

两者都占 5001，同时只能跑一个：

```bash
docker compose -f docker/docker-compose.asr.yml stop audio-inferer
docker compose -f docker/docker-compose.asr.yml --profile nemo up -d audio-inferer-nemo
```

同时把 `pipelines/asr-server/config.yml` 里 `AudioInferer.backend` 改为 `nemo`。

### 本地 vLLM VLM 后端

```bash
VLLM_VLM_MODEL=openbmb/MiniCPM-V-2_6 \
docker compose -f docker/docker-compose.vfa.yml --profile mllm up -d
```

frame analyzer 容器内已配置 `host.docker.internal` → 宿主机，config 里的
`http://localhost:8000/v1` 需改为 `http://host.docker.internal:8000/v1`
（或直接用 compose 服务名 `http://vllm-vlm:8000/v1`）。

## 挂载说明

- `pipelines/asr-server` / `pipelines/vfa-server` → 容器 `/project`：
  config.yml、temp/、runtime 日志都落在宿主机，行为和 conda 方式一致
- 模型缓存（HuggingFace / torch hub / ModelScope）用 named volume 共享，
  容器重建不用重新下载
- `~/.openmmla` 只读挂载：容器内可解密 ENC(...) 密钥

## 与 TUI 的关系

TUI Launcher 中 ASR Server / VFA Server 的 Start / Stop / Logs 已全部改为
docker compose（tmux+gunicorn 方式已移除）：

- **Start**：`docker compose -f docker/docker-compose.*.yml up -d --build <选中的子服务>`
  （AudioInferer 按 config 的 `backend` 自动选 wespeaker 或 nemo 容器）
- **Stop**：`docker compose ... down`
- **Logs**：`docker compose ... logs --tail 40`
- **状态**：仍按端口探测，容器起来即显示 [OK]

远程 Host 走 SSH 在远端仓库目录执行同样的命令，因此远程机器需要：
仓库已拉取（含 docker/ 目录）、Docker Engine + nvidia-container-toolkit、
当前用户在 docker 组（无需 sudo 运行 docker）。

## 已知注意点

- nemo 镜像标记为实验性：nemo-toolkit ≤1.23 依赖较老，构建时间长
- transcriber 镜像用 whisperx 3.8.6（语言、模型名等 config 不变：`whisperx/small`、`da`）；
  首次请求会下载模型到 hf-cache volume
- separator 的 MossFormer2 模型缓存写在 `/root/.cache/modelscope`（named volume）
