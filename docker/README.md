# Dockerized OpenMMLA services (ASR / VFA / 中心基础设施)

One image per service, so每个服务的 Python 环境完全隔离，升级互不影响。
端口与 nginx 网关的 upstream 完全一致，网关配置无需改动。

下表是 AI 服务；中心基础设施（InfluxDB / MongoDB）见
[中心基础设施 (InfluxDB / MongoDB)](#中心基础设施-influxdb--mongodb)。

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

中心基础设施（`docker-compose.infra.yml`）只需要 Docker Engine：不用 GPU、
不用 nvidia-container-toolkit、也不需要 master key。

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

## 中心基础设施 (InfluxDB / MongoDB)

`docker-compose.infra.yml` 把 Uber Server 的两个数据库跑成容器，替代 brew /
apt 的裸机安装。默认端口与 `config/system_services.yml` 一致，pipeline 只需改主机名；
宿主机端口可以用 `INFLUXDB_PORT` / `MONGODB_PORT` 改掉（见下面「和别的项目共用一台机器」）。

| Service | Image | Port | GPU | Stack |
|---|---|---|---|---|
| InfluxDB | `influxdb:2.7.12` | 8086 | — | 官方镜像，v2 API（org/bucket/token + Flux） |
| MongoDB | `mongo:7.0.40-jammy` | 27017 | — | 官方镜像，默认不开认证 |

镜像标签是钉死的，不要换成 `latest`：`influxdb:latest` 在 2026-09-15 会指向
InfluxDB 3 Core（没有 org/bucket/token 那套语义，`influxdb-client==1.44.0` 直接失效），
`mongo:latest` 则会跟着大版本漂移。两个标签都同时发布 linux/amd64 和 linux/arm64。

在**要放数据库的那台机器**（下文假设主机名 `server-01`）的仓库根目录执行。
这台机器只需要拉一份仓库（或者干脆只拷贝 `docker/` 下这两个文件，本 stack
不需要 build context）。

先停掉可能占着 8086 / 27017 的裸机服务，否则 `up -d` 会以
`Bind for 0.0.0.0:8086 failed: port is already allocated` 失败（端口被**别的项目的
容器**占着的情况见下面「和别的项目共用一台机器」，那种不能停，要换端口）：

```bash
# macOS
brew services stop influxdb mongodb-community
# Ubuntu/Debian
sudo systemctl disable --now influxdb mongod
```

然后填密钥并启动：

```bash
cp docker/.env.example docker/.env && chmod 600 docker/.env
# 编辑 docker/.env，至少填 INFLUXDB_INIT_ADMIN_TOKEN 和 INFLUXDB_INIT_PASSWORD
#   openssl rand -hex 32      → INFLUXDB_INIT_ADMIN_TOKEN
#   openssl rand -base64 24   → INFLUXDB_INIT_PASSWORD

docker compose -f docker/docker-compose.infra.yml up -d

# 状态 / 日志 / 停止（down 保留 volume，down -v 会删光数据）
docker compose -f docker/docker-compose.infra.yml ps
docker compose -f docker/docker-compose.infra.yml logs -f influxdb
docker compose -f docker/docker-compose.infra.yml down
```

密钥请写进 `docker/.env`（已被 `.gitignore` 忽略），不要用 `export`：compose
读取 `.env` 是对**每一条子命令**生效的，换个 shell 或重启之后 `ps` / `logs` /
`down` 行为一致；`export` 出来的值只活在当前那个 shell 里。`export` 还会把 token
留在 `~/.bash_history` 里。`INFRA_BIND_ADDRESS` 同理——写进 `.env` 才能保证之后每次
`up -d` 都用同一个绑定地址，否则会悄悄退回 `0.0.0.0`。

### 从裸机数据库迁移过来（先做这一步）

容器起来时两个数据库都是**全新的空库**。直接改 URL 指过去，TUI 的 session 列表、
dashboard 的历史 session、以及全部历史 `sensor_events` 都会看起来「消失」——
数据还在旧机器上，只是没人再连它了。

要保留历史数据，先迁移再改 URL。在**旧的** Uber Server（比如 `ericli.local`）上导出：

```bash
mongodump --uri "mongodb://localhost:27017" --db openmmla --archive=openmmla.archive
influx backup ./influx-backup -t "<旧的 admin token>"
```

拷到 server-01 之后导入容器：

```bash
docker compose -f docker/docker-compose.infra.yml exec -T mongodb \
  mongorestore --archive --db openmmla < openmmla.archive

docker cp ./influx-backup "$(docker compose -f docker/docker-compose.infra.yml ps -q influxdb)":/tmp/influx-backup
docker compose -f docker/docker-compose.infra.yml exec influxdb \
  influx restore /tmp/influx-backup --full
```

`influx restore --full` 会连同 token 一起覆盖成旧实例的，之后 TUI 里填**旧
token**；不想覆盖就用 `--bucket mmla-data` 只恢复数据，token 用新的。

不想迁移也可以：把旧机器继续开着，只是新旧数据从此分家。

### 备份

数据只存在 named volume 里，`docker compose down -v`、`docker volume rm`、
以及某些「重置一下」的教程命令都会**永久删除**它们，没有回收站。
至少定期跑一次上面那两条导出命令，落到 Docker 之外的路径：

```bash
docker compose -f docker/docker-compose.infra.yml exec -T mongodb \
  mongodump --archive --db openmmla > /backup/openmmla-$(date +%F).archive

docker compose -f docker/docker-compose.infra.yml exec influxdb \
  influx backup /tmp/backup -t "<token>"
docker cp "$(docker compose -f docker/docker-compose.infra.yml ps -q influxdb)":/tmp/backup /backup/influx-$(date +%F)
```

采集进行中不要 `down`；两个服务都设了 `stop_grace_period: 60s`，但正常流程是
先结束 session 再停 stack。

### 和别的项目共用一台机器

同一台机器上如果已经有别的项目的 InfluxDB / MongoDB 容器（哪怕它们只绑
`127.0.0.1:8086`），我们的 `0.0.0.0:8086` 也起不来——内核不允许同一端口上
通配地址和具体地址并存。**不要去停别人的容器**，改我们的宿主机端口：

```bash
# docker/.env
INFLUXDB_PORT=8087
MONGODB_PORT=27018
```

然后 System Services 里的 url 带上新端口（`http://server-01:8087`、
`mongodb://server-01:27018`）。TUI 卡片的状态探测会从 url 里读端口，不用改代码。
容器内部仍是 8086 / 27017，healthcheck 和数据都不受影响。

不要反过来"共用"别人的实例：它多半只绑 loopback（别的机器连不上）、开了认证
（密码得明文进 `MongoDB.url`），而且 `influxdb:latest` 在 2026-09-15 之后一次
`pull` 就会变成 InfluxDB 3，把你的数据一起带进坑里。

### 让 OpenMMLA 指向这套基础设施

`mmla tui` → Launcher → System Services，只改这三处：

| 字段 | 值 |
|---|---|
| `InfluxDB.url` | `http://server-01.local:8086`（`org: admin`、`bucket: mmla-data` 保持不变） |
| `InfluxDB.token` | 上面那个 `INFLUXDB_INIT_ADMIN_TOKEN`，把原来的 `ENC(...)` 整串替换成明文 |
| `MongoDB.url` | `mongodb://server-01.local:27017`（`db: openmmla` 不变） |

**主机名怎么写取决于你的网络。** `.local` 是 mDNS，只在**同一个局域网**里有效。如果你的
Mac 和 server-01 不在一个网段、中间走的是 Tailscale（`ssh admin@server-01` 能通、
但 `ping server-01.local` 不通就是这种情况），要写 Tailscale 的 MagicDNS 名字或
tailnet IP：`http://server-01:8086`、`http://100.x.x.x:8086`。这也意味着**每一台跑
base station 的机器都得在 tailnet 里**，否则它到不了数据库。改了端口的话把端口
一起写上：`http://server-01:8087`。

保存时 token 会自动重新加密成 `ENC(...)`，并同步写回所有本地
`pipelines/*/config.yml`，**不要手改那些文件**（会被覆盖，运行时也以
`config/system_services.yml` 为准）。远程 Host 的第一次 launch 会先同步 config
再要求重新 launch，这是设计如此，不是报错。

新容器是全新的 InfluxDB，旧的 token 一定认证失败，必须换成新 token。

验证顺序（从 Mac 上执行）：

```bash
ping -c1 server-01.local
curl -sf http://server-01.local:8086/health
mongosh mongodb://server-01.local:27017 --eval 'db.adminCommand({ping:1})'
```

`server-01.local` 依赖 mDNS：Ubuntu Server 默认既没有 avahi-daemon 也没有
libnss-mdns（`sudo apt install -y avahi-daemon libnss-mdns` +
`sudo hostnamectl set-hostname server-01`），且 mDNS 不跨子网 / VLAN。
解析不了就退回固定 IP 或 `/etc/hosts`。

**注意这三条验证命令是在宿主机上跑的，但真正读这些 URL 的还有 ASR / VFA 容器**，
而 bridge 网络里的容器默认不做 mDNS 解析——宿主机 `ping server-01.local` 通，
不代表容器里通。同时要用容器化 AI 服务的话，`config/system_services.yml` 里
建议直接填固定 IP，或者给那两个 compose 文件加 `extra_hosts`。

### MongoDB 认证（可选，强烈建议）

默认不开认证，和现在的裸机部署一致。要开就在**第一次启动前**同时设置两个变量：

```bash
export MONGO_ROOT_USER=openmmla
export MONGO_ROOT_PASSWORD="<密码>"
docker compose -f docker/docker-compose.infra.yml up -d
```

- 只在 `mongodb-data` volume 为空时生效；volume 里已经有数据之后再加变量，
  会变成「开了 `--auth` 但没有任何用户」，谁都连不上。**这时不要删 volume**：
  把两个变量清空再 `up -d` 就回到无认证状态、数据分毫不动；或者用容器内的
  localhost exception 建用户：
  `docker compose -f docker/docker-compose.infra.yml exec mongodb mongosh admin --eval 'db.createUser({user:"openmmla",pwd:"<密码>",roles:["root"]})'`
- 只设置其中一个变量，容器会**反复重启**，而 `up -d` 依然报告成功。
  27017 一直不通就查 `docker compose -f docker/docker-compose.infra.yml ps` 和
  `... logs mongodb`
- 开了之后 URL 必须写成
  `mongodb://<user>:<pass>@server-01.local:27017/?authSource=admin`，
  漏掉 `authSource=admin` 会认证失败
- 但要注意：`url` 不在加密字段名单里（只有 token/password/secret 这类 key 会加密），
  而 `config/system_services.yml` 是**被 git 跟踪的**，所以带密码的 URL 会以明文提交。
  在给 `MongoDB` 增加独立的 username/password 字段之前，先权衡这一点

## 挂载说明

- `pipelines/asr-server` / `pipelines/vfa-server` → 容器 `/project`：
  config.yml、temp/、runtime 日志都落在宿主机，行为和 conda 方式一致
- 模型缓存（HuggingFace / torch hub / ModelScope）用 named volume 共享，
  容器重建不用重新下载
- `~/.openmmla` 只读挂载：容器内可解密 ENC(...) 密钥
- 基础设施的数据全在 named volume 里：`influxdb-data`（`/var/lib/influxdb2`，
  含 influxd.bolt 与 engine）、`influxdb-config`（`/etc/influxdb2`，含
  `influx-configs`，admin token 可从这里找回）、`mongodb-data`（`/data/db`）、
  `mongodb-config`（`/data/configdb`）。不要用 bind mount 挂 `/data/db`，
  WiredTiger 需要真实的文件锁语义
- 基础设施容器**不**挂 `~/.openmmla`：官方 influxdb / mongo 镜像里没有 openmmla
  代码，也不会读 config.yml，没有可解密的东西

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

中心基础设施目前**没有**接进 TUI，`docker-compose.infra.yml` 只能手动
`docker compose` 起停：

- **状态**：`Uber: InfluxDB` / `Uber: MongoDB` 卡片探的是 System Services 里配的
  url（从 TUI 这台机器直接 TCP 连 host:port），和 Host 选择器无关；只有 url 写成
  localhost 时才退回"探选中主机自己的回环"。Status 页同理，端口列会显示实际
  探测的 host:port
- **Start / Stop / Logs**：卡片上有一个 **Run mode** 下拉（`docker` / `native`），
  **默认 `docker`**。还在用 brew / systemctl 裸机数据库的机器把它切回 `native`，
  否则 Start 会在那台机器上起容器、和裸机实例抢端口。`docker` 模式下三个按钮走
  `docker compose -f docker/docker-compose.infra.yml up -d / stop / logs <服务>`。
  停止用 `stop` 而不是 `down`：两张卡片共用一个 compose 文件，`down` 会把另一个
  数据库容器一起拆掉
- Run mode 是按「主机 + 服务」记住的，切换 Host 或点别的节点再回来不会丢。但它
  只存在这次 TUI 会话里，重启 TUI 会回到默认的 `docker`
- 状态探测**直接连 System Services 里配的 host:port**，从跑 TUI 的这台机器发起——
  也就是 pipeline 真正走的那条路。`InfluxDB.url` 写 `http://server-01:8087`，卡片和
  Status 页就去连 `server-01:8087`，`INFRA_BIND_ADDRESS` 绑在哪张网卡都无所谓。
  卡片描述里会写出探的是哪个地址。两个推论：
  - url 是 `localhost` / `127.0.0.1` 时它不指向任何一台特定机器，这时沿用老逻辑：
    本地探本机回环，远程 Host 走 SSH 探那台机器自己的回环
  - "可达"是**从 TUI 这台机器看**的。TUI 机器不在 tailnet 里、或者被防火墙挡着，
    卡片会灰，哪怕 pipeline 机器连得上
- 数据库搬到 server-01 之后，Mac 本地的这两张卡片会一直显示未运行（本地探测写死
  127.0.0.1），这是预期现象，不是连不上。**这时更不要在 Mac 上点 Start**：
  按钮走的是 `make influxdb` / `make mongodb`，会在本机 8086 / 27017 起一个裸机
  数据库，卡片随即变绿 [OK]——但那是一个空的本地库，pipeline 连的仍然是
  server-01，绿灯反而会误导。搬完之后把 Mac 上的这两个裸机服务也停掉

## 已知注意点

- nemo 镜像标记为实验性：nemo-toolkit ≤1.23 依赖较老，构建时间长
- transcriber 镜像用 whisperx 3.8.6（语言、模型名等 config 不变：`whisperx/small`、`da`）；
  首次请求会下载模型到 hf-cache volume
- separator 的 MossFormer2 模型缓存写在 `/root/.cache/modelscope`（named volume）
- **`DOCKER_INFLUXDB_INIT_*` 只在第一次启动、且 `influxdb-data` 为空时生效**。
  entrypoint 见到 `influxd.bolt` 就整个跳过 setup，之后改 token / org / bucket /
  密码再 `up -d` 不报错也不生效。事后换 token 要用
  `docker compose -f docker/docker-compose.infra.yml exec influxdb influx auth create --org admin --all-access`；
  找回初始 token 用 `... exec influxdb cat /etc/influxdb2/influx-configs`
- **不要为了重新 bootstrap 去删 `influxd.bolt`**。entrypoint 只用它是否存在来判断
  要不要跑 setup；删掉之后 setup 会重跑，而 `/etc/influxdb2` 是持久化的、里面已经
  有配置，setup 大概率失败，其失败路径会对 engine 目录执行 `rm -rf`——数据全没。
  要换 token 用上一条的 `influx auth create`
- InfluxDB 的 Web UI 同样发布在 8086 上，任何能访问这个端口的人都能用
  `admin` + `INFLUXDB_INIT_PASSWORD` 登进去。这个密码别用 8 位凑数，
  用 `openssl rand -base64 24` 生成
- **在跑容器化数据库的机器上不要执行 `make -C pipelines/uber-server all`**：
  `all` 先跑 `clean-ports`，会对占用 8086 / 27017 的进程 `kill -9`，也就是
  docker-proxy；随后的 `influxdb` / `mongodb` target 还会去启动裸机服务。
  这类机器上请用 `make all without=influxdb,mongodb`
- MongoDB 默认无认证，且在 Linux 上 Docker 的 NAT 规则先于 ufw 生效，
  `ufw deny 27017` 挡不住已发布的端口。只在可信实验室网段里用，或者用
  `INFRA_BIND_ADDRESS=<内网 IP>` 把发布限制在一张网卡上，
  再不然写 `DOCKER-USER` 链的 iptables 规则
- MongoDB 5.0+ 要求 x86_64 有 AVX 指令集（arm64 要求 ARMv8.2-A 以上）。
  老 CPU 上容器会以 exit 132 反复重启，`up -d` 却是「成功」的。上线前先在
  server-01 上跑 `grep -m1 -o avx /proc/cpuinfo`
