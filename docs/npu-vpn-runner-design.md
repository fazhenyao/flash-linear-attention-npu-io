# NPU VPN 性能采集任务执行方案

## 1. 文档目的

本文设计一套从性能看板提交 `msprof` / `msopprof` 采集任务，并在只能通过 VPN 访问的 NPU 服务器上可靠执行、回传指标和保留制品的方案。

推荐架构为：

> Cloudflare Worker/D1 负责任务控制面，VPN Runner Relay 负责跨越 VPN，NPU 服务器负责实际执行，大体积采集制品保存在 NPU 服务器或 Relay 本地磁盘。

当前部署不启用 R2。文中涉及 R2 或签名下载的内容属于阶段三可选扩展，不是当前闭环的组成部分。

本文同时记录目标架构和当前实现。标注为“阶段二”或“建议”的 systemd 持久执行、远端 Device 锁和断线恢复尚未落地；当前闭环采用 Windows Relay 同步 SSH 执行和本地制品保留。

该方案不要求 NPU 服务器暴露公网端口，也不要求 Cloudflare Worker 直接连接 SSH。

最终安全决策为：即使 Relay 具备公网可达条件，也不把 Relay 建设为公网服务。Relay 不提供 Webhook、不运行公网 HTTP API、不开放公网监听端口；任务领取、状态回传和制品上传全部由 Relay 主动发起出站连接。

## 2. 当前条件

- 性能看板部署在 GitHub Pages，可访问 Cloudflare Worker。
- Cloudflare Worker 已连接 D1，并提供用户认证和权限控制。
- `backend/perf_runner.py` 已支持 `PERF_RUN_MODE=ssh` 和 `PERF_RUN_MODE=local`。
- `msprof` / `msopprof` 必须运行在安装了 NPU 驱动和工具链的服务器上。
- NPU 服务器只能在本机或指定网关开启 VPN 后通过 SSH 访问。
- VPN 可能断开、续期、需要 MFA，不能假设连接永久稳定。
- Profiling 原始目录可能很大，不适合直接存入 D1。
- 当前 Relay 是用户开启 VPN 的 Windows 电脑，通过两个登录触发的计划任务分别运行 A2 和 A5 Agent。
- A2 SSH 目标为 `root@192.168.9.221:22`，项目目录为 `/workspace/fazhenyao/flash-linear-attention-npu-io`；CANN 环境脚本为 `/data/fazhenyao/cann/3_23/ascend-toolkit/set_env.sh`，Conda 环境为 `fla_dump`。
- A5 SSH 目标为 `npu_user7@192.168.13.241:22`，采集输出工作目录为 `/home/npu_user7/fazhenyao/flash-linear-attention-npu-io`，执行脚本为 `/home/npu_user7/fazhenyao/flash-linear-attention-npu/examples/flash_gated_delta_rule.py`；CANN 环境脚本为 `/home/npu_user7/fazhenyao/cann/7_20/ascend-toolkit/set_env.sh`，Conda 环境为 `f30077529`。

## 3. 设计目标

1. 用户可以在公网性能看板提交采集任务并查看进度。
2. NPU 服务器无需开放公网入站端口。
3. Relay 无需开放公网入站端口，不直接承受公网扫描、暴力请求和 DDoS。
4. VPN 未连接时任务可靠排队，不丢失、不误报失败。
5. VPN 执行中断开时，远端命令尽可能继续执行并可恢复跟踪。
6. 防止同一任务或同一 NPU Device 被重复并发执行。
7. 用户不能通过任务参数执行任意 Shell 命令。
8. 结构化指标可直接在看板展示，大文件通过 VPN/SSH 由管理员获取。
9. 每次执行可追踪到用户、代码版本、运行环境、命令模板和制品。

## 4. 非目标

- Cloudflare Worker 不直接执行 `msprof` / `msopprof`。
- Cloudflare Worker 不直接建立 SSH 或 VPN 连接。
- GitHub Pages 不直接访问本机 `127.0.0.1` 或 NPU 服务器。
- Relay 不接收 Cloudflare Worker 或其他公网系统发起的 Webhook。
- Relay 不通过公网 HTTP API 接收采集任务。
- D1 不存储完整 Prof 目录、压缩包或超大日志。
- 用户提交的数据中不允许包含任意可执行命令。

## 5. 推荐架构

```mermaid
flowchart LR
    U[性能看板] -->|提交与查询任务| W[Cloudflare Worker]
    W <--> D[(Cloudflare D1)]
    R[VPN Runner Relay] -->|出站 HTTPS 领取任务/回传结果| W
    R -->|检查并建立| V[VPN]
    V -->|SSH / SCP| N[NPU 服务器]
    N -->|msprof / msopprof| P[Prof 结果目录]
    P -->|SCP 回收新增目录| R
    R -->|结构化结果与制品清单| W
    R -->|本地保留 30 天| L[(Relay / NPU 本地磁盘)]
```

### 5.1 组件职责

#### 性能看板

- 提交结构化采集参数。
- 展示任务状态、排队原因、执行日志摘要和结构化指标。
- 提供取消、重试和结构化结果查看入口。
- 不持有 Runner 凭据和 SSH 凭据。

#### Cloudflare Worker

- 验证用户身份和提交权限。
- 校验任务类型、脚本、参数范围和幂等键。
- 管理任务状态机、Agent 注册、租约、事件和取消请求。
- 为看板提供查询接口，为 Relay 提供专用接口。
- 保存结构化结果和制品元数据。
- 不执行长时间采集命令。

#### Cloudflare D1

- 保存任务、状态、执行尝试、Agent、事件、结果和制品元数据。
- 作为控制面的事实来源。
- 不保存完整 stdout/stderr 和 Prof 压缩包。

#### VPN Runner Relay

- 部署在同时能访问互联网和 VPN 的机器上。
- 主动通过 HTTPS 从 Worker 领取任务，不开放公网监听端口。
- 领取前检查 NPU SSH 端口可达性；执行时由 SSH 命令加载并验证远端环境。
- 复用 `backend/perf_runner.py` 的命令构建和结果导入逻辑。
- 通过同步 SSH 命令执行采集，通过 SCP 回收本次新增的结果目录。
- 发送 heartbeat、进度和日志摘要。
- 在本地保留制品并回传结构化结果、路径、大小和 SHA-256。

#### 当前 NPU 服务器执行方式

- Relay 进入固定项目目录并执行服务端白名单生成的命令，不接收用户提供的任意 Shell。
- 每次命令先加载 CANN `set_env.sh`，再加载 Conda 初始化脚本并激活 `fla_dump`。
- `msprof` 输出到 `data/prof_gdr`，`msopprof` 输出到 `data/prof_op`。
- 当前 SSH 会话同步等待采集完成；`systemd-run`、远端状态文件和 Device 锁属于阶段二。

#### 本地制品存储

- NPU 服务器或 Relay 保存 Prof 压缩包、CSV、JSON、图片和完整日志。
- 制品目录不得通过公网服务暴露，只允许管理员经 VPN/SSH 访问。
- Relay 回收副本默认保留 30 天，由 Agent 清理；当前不会自动删除 NPU 服务器上的源目录。
- D1 只记录制品路径、类型、大小、SHA-256 和过期时间。

#### 可选 Cloudflare R2

- 仅在需要浏览器在线下载制品时启用。
- Bucket 必须保持私有，下载使用短期签名地址。
- 启用前需要完成计费、容量上限和生命周期策略评审。

### 5.2 网络与攻击面

```text
性能看板  --公网 HTTPS--> Cloudflare Worker / D1
Relay     --公网 HTTPS--> Cloudflare Worker
Relay     --VPN 内 SSH--> NPU 服务器

互联网    --禁止-------> Relay
互联网    --禁止-------> NPU 服务器
Worker    --不连接-----> Relay
```

Relay 主机防火墙默认拒绝所有公网入站连接。Relay 不配置公网域名，不运行 Cloudflare Tunnel，也不部署 Webhook 服务。公网扫描和 DDoS 攻击面集中在 Cloudflare Worker；即使 Worker 暂时不可用，影响也应限制为任务延迟，不能形成进入 Relay 或 VPN 内网的连接路径。

Relay 只允许以下必要出站流量：

- 到 Worker API 的 HTTPS 443，用于领取任务、heartbeat 和回传结果。
- 通过 VPN 到 NPU 服务器的 SSH 22，用于执行和回收结果。

可在 Relay 上进一步设置出站白名单、DNS 限制和响应体大小限制，减少 Worker 凭据泄露或异常响应造成的风险。

## 6. 部署拓扑选择

### 6.1 推荐：专用 VPN Relay

在一台长期在线的专用机器上安装 VPN 客户端和 Relay：

```text
互联网访问：允许访问 Worker 的 HTTPS 443
VPN 访问：允许连接内部网络
NPU 访问：VPN 建立后允许 SSH 22
运行方式：Linux systemd、Windows Service 或 Windows 计划任务
```

优点是任务不依赖个人电脑是否在线，VPN 凭据和 SSH 密钥也可以集中治理。

专用 Relay 即使拥有公网 IP，也应关闭所有非必要入站服务。运维登录应通过受控管理网络或 VPN 进行，不应为了接收任务开放公网 HTTP 端口。

### 6.2 过渡方案：个人电脑 Relay

如果当前 VPN 只能在个人电脑上运行，可先在该电脑启动 Relay：

- 电脑或 VPN 离线时，任务保留在 D1 中等待。
- Relay 上线后自动消费队列。
- 看板显示“等待 VPN 执行器上线”。
- 不应把它作为无人值守生产方案。

### 6.3 可选：NPU 服务器直接运行 Agent

如果 NPU 服务器能够主动访问互联网，可直接在 NPU 服务器运行 Agent，省略 SSH 和 Relay。Agent 仍然主动连接 Worker，不开放公网端口。

如果 NPU 服务器无法访问互联网，则继续使用 VPN Relay。

## 7. 任务执行流程

```mermaid
sequenceDiagram
    participant UI as 性能看板
    participant W as Worker
    participant D as D1
    participant R as VPN Relay
    participant N as NPU Server

    UI->>W: POST /api/perf/jobs
    W->>D: INSERT queued job
    W-->>UI: job_id

    loop 任务领取
        R->>R: 检查 NPU SSH 端口可达并发送 Runner heartbeat
        R->>W: POST /api/runner/jobs/claim
        W->>D: 条件更新 queued -> claimed
        W-->>R: job + lease_token
    end

    R->>W: POST /started
    R->>N: SSH 列出执行前 PROF_* / OPPROF_*
    R->>N: SSH 加载 CANN/Conda 并同步执行 profiler
    Note over R,W: SSH 执行期间独立线程每 15 秒发送 job heartbeat
    N-->>R: SSH 命令退出
    R->>N: SSH 识别本次新增结果目录
    R->>N: SCP 回收完整结果目录
    R->>R: 解析指标，计算目录大小和 SHA-256
    R->>W: POST /complete，回传结构化结果和制品元数据
    W->>D: 保存结果和制品元数据
    UI->>W: GET /api/perf/jobs/{job_id}
    W-->>UI: succeeded + metrics + artifact manifest
```

### 7.1 自适应任务领取

当前 Relay 使用线性自适应轮询，配置为 `RUNNER_POLL_MIN_SECONDS=2`、`RUNNER_POLL_MAX_SECONDS=30`：

```text
Relay 启动：注册后立即执行一次 heartbeat、健康检查和领取，不先等待
连续未领取到任务：等待时间按 4、6、8……30 秒递增
VPN 或 SSH 不可用：本轮不领取，沿用相同线性退避，最大 30 秒
成功领取并完成任务：下一轮等待重置为 2 秒
任务执行中：单并发，不领取新任务；独立线程每 15 秒发送 job heartbeat
Worker 请求异常：至少等待 5 秒并指数退避，最大 120 秒
```

当前实现没有随机抖动，也没有采用 Worker 返回的 `retry_after_seconds`。部署多个 Relay 时，应增加抖动并尊重服务端建议间隔，避免同时集中请求。

领取接口只返回 Relay 能力范围内的任务，并限制响应数量和响应体大小。没有可执行任务时返回正常空结果，不把空队列作为错误。

### 7.2 多 Relay 精确路由

看板从 Worker 的 Runner 状态接口读取在线 Agent，并在“执行服务器”下拉框中显示 Runner 名称、芯片、默认 NPU 卡号和在线状态。提交任务时同时发送：

- `target_runner_id`：指定本次任务必须由哪个 Agent 领取。
- `chip`、`device` 和 `prof_tool`：继续作为能力匹配和执行参数。

Worker 领取任务时先校验 `target_runner_id`，再校验 Agent 上报的芯片、卡号和 profiler 能力。未被指定的 Agent 即使芯片相同也不能领取该任务；目标 Agent 离线时，任务保留在 D1 等待，不会自动漂移到其他服务器。

当前两个 Agent 共享同一个 Runner Token，但使用不同的 `RUNNER_ID`、配置文件、状态目录、日志、计划任务和本地制品目录。后续应为每个 Agent 签发可单独吊销和轮换的凭据。

任务提交仍只接受 Worker 白名单中的 `script_id`。`PERF_REMOTE_SCRIPT` 是受信任的 Relay 本机配置，用于把该白名单 ID 映射到特定 NPU 主机上的脚本路径；浏览器用户不能提交或覆盖远端路径。

## 8. 任务状态机

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> waiting_runner
    waiting_runner --> queued: Agent 可用
    queued --> waiting_vpn
    waiting_vpn --> queued: VPN/SSH 恢复
    queued --> claimed
    waiting_runner --> claimed
    waiting_vpn --> claimed
    claimed --> running
    running --> succeeded
    claimed --> failed
    running --> failed
    running --> disconnected: VPN/网络中断
    disconnected --> running: 恢复并确认仍在执行
    disconnected --> succeeded: 恢复并确认已完成
    disconnected --> orphaned: 无法确认远端状态
    queued --> canceled
    claimed --> cancel_requested
    running --> cancel_requested
    cancel_requested --> canceled
    failed --> queued: 重试
    canceled --> queued: 重试
    orphaned --> queued: 人工确认后重试
```

### 8.1 断线语义

- 领取前 VPN 不可用：不领取任务，记录 `waiting_vpn`。
- 领取后尚未远端启动：安全释放任务并重新排队。
- 远端启动后 VPN 断开：Worker 可标记 `disconnected`，但当前同步 SSH 实现通常会因 SSH 失败而结束本次尝试。
- VPN 恢复后按 `job_id` 查询远端状态并继续跟踪属于阶段二；当前 Agent 尚未实现自动 reconcile。
- 无法确认远端状态：进入 `orphaned`，禁止自动重跑。
- 只有确认旧进程不存在或已结束，才允许人工或自动重新执行。

不能简单地在 heartbeat 超时后重新排队，因为原任务可能仍在 NPU 上运行，重新执行会产生重复任务和 Device 冲突。当前版本在执行期断线后应由管理员确认远端进程和 Prof 目录，再决定是否重试。

## 9. API 设计

### 9.1 用户接口

```text
POST /api/perf/jobs
GET  /api/perf/jobs
GET  /api/perf/jobs/{job_id}
GET  /api/perf/jobs/{job_id}/events
POST /api/perf/jobs/{job_id}/cancel
POST /api/perf/jobs/{job_id}/retry
GET  /api/perf/jobs/{job_id}/artifacts
```

### 9.2 Runner 接口

```text
POST /api/runner/register
POST /api/runner/heartbeat
POST /api/runner/jobs/claim
POST /api/runner/jobs/{job_id}/started
POST /api/runner/jobs/{job_id}/heartbeat
POST /api/runner/jobs/{job_id}/events
POST /api/runner/jobs/{job_id}/artifacts
POST /api/runner/jobs/{job_id}/complete
POST /api/runner/jobs/{job_id}/fail
POST /api/runner/jobs/{job_id}/reconcile
```

Runner 接口使用独立的服务凭据，不能复用浏览器用户 Token。

阶段三启用对象存储后，再增加制品上传和签名下载接口；当前接口只登记本地制品清单。

### 9.3 用户提交示例

```json
{
  "tool": "msprof_op",
  "script_id": "scripts/flash_gated_delta_rule.py",
  "case_id": "case-kda-h96",
  "chip": "A2",
  "device": 2,
  "kernel_name": "kda_forward",
  "parameters": {
    "batch": 1,
    "tokens": 4096,
    "query_heads": 32,
    "value_heads": 32,
    "key_dim": 128,
    "value_dim": 128,
    "chunk_size": 64,
    "dtype": "bf16"
  },
  "idempotency_key": "4eae9183-c018-4d5e-a455-d5c5bc393043"
}
```

Worker 应将用户参数转换为规范化任务，不允许提交 `command`、Shell 片段或任意脚本路径。

### 9.4 完成结果示例

```json
{
  "attempt_id": "attempt-9e2f",
  "lease_token": "runner-issued-lease-token",
  "exit_code": 0,
  "started_at": "2026-08-08T10:00:00+08:00",
  "finished_at": "2026-08-08T10:08:32+08:00",
  "environment": {
    "agent_version": "1.0.0",
    "git_commit": "b78596cab5ea312c11bb21bdb5f2d4490fe1cb80",
    "host": "npu-a2-01",
    "chip": "A2",
    "device": 2,
    "profiler": "msopprof"
  },
  "metrics": {
    "duration_us": 131.24,
    "mfu": 0.61,
    "mbu": 0.73
  },
  "artifacts": [
    {
      "type": "prof_archive",
      "object_key": "perf/jobs/job-123/prof.tar.zst",
      "size": 104857600,
      "sha256": "..."
    }
  ]
}
```

## 10. D1 数据模型

### 10.1 perf_jobs

```sql
CREATE TABLE perf_jobs (
  id TEXT PRIMARY KEY,
  created_by TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  tool TEXT NOT NULL,
  script_id TEXT NOT NULL,
  request_json TEXT NOT NULL,
  status TEXT NOT NULL,
  status_message TEXT NOT NULL DEFAULT '',
  runner_id TEXT,
  attempt_id TEXT,
  lease_token_hash TEXT,
  lease_expires_at TEXT,
  remote_execution_id TEXT,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  exit_code INTEGER,
  created_at TEXT NOT NULL,
  claimed_at TEXT,
  started_at TEXT,
  finished_at TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE(created_by, idempotency_key)
);
```

### 10.2 perf_job_events

保存状态变化、有限长度日志和诊断信息。完整日志保存在 NPU 服务器或 Relay 本地磁盘。

```sql
CREATE TABLE perf_job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  attempt_id TEXT,
  event_type TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info',
  message TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
```

### 10.3 perf_results

保存看板需要查询和比较的结构化指标。

```sql
CREATE TABLE perf_results (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL UNIQUE,
  case_id TEXT,
  model_id TEXT,
  snapshot_id TEXT,
  environment_json TEXT NOT NULL DEFAULT '{}',
  metrics_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
```

### 10.4 perf_artifacts

```sql
CREATE TABLE perf_artifacts (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  object_key TEXT NOT NULL,
  filename TEXT NOT NULL,
  content_type TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

### 10.5 runner_agents

```sql
CREATE TABLE runner_agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  capabilities_json TEXT NOT NULL DEFAULT '{}',
  vpn_connected INTEGER NOT NULL DEFAULT 0,
  npu_reachable INTEGER NOT NULL DEFAULT 0,
  last_heartbeat_at TEXT,
  updated_at TEXT NOT NULL
);
```

## 11. 原子领取和租约

多个 Relay 可能同时看到同一个排队任务。领取必须使用带状态条件的更新：

```sql
UPDATE perf_jobs
SET status = 'claimed',
    runner_id = ?,
    attempt_id = ?,
    lease_token_hash = ?,
    lease_expires_at = ?,
    claimed_at = ?,
    updated_at = ?
WHERE id = ?
  AND status = 'queued'
  AND cancel_requested = 0;
```

只有更新行数为 1 的 Relay 获得任务。后续 Runner 请求必须同时携带 `job_id`、`attempt_id` 和 `lease_token`。

租约用于防止失联 Agent 永久占用任务，但不能在远端命令启动后直接触发自动重跑。Worker 需要根据 `remote_execution_id` 进入恢复或人工确认流程。

## 12. VPN 和 SSH 健康检查

当前 Relay 领取任务前执行以下检查：

1. 向 Worker 发送 Runner heartbeat，确认控制面 HTTPS 可用。
2. 加载本地 Relay 配置并确认执行模式为 SSH。
3. 使用 5 秒 TCP 超时探测 `192.168.9.221:22`；不可达时不领取任务。
4. 领取后执行真正的 SSH 命令；SSH/SCP 使用 `BatchMode=yes`、`ConnectTimeout=10`、`ServerAliveInterval=15` 和 `ServerAliveCountMax=4`。

当前端口探测不能证明 SSH 密钥、CANN、Conda、profiler、Device 空闲状态或磁盘空间正常。后续应把 SSH 认证、`npu-smi`、profiler 可用性、Device 锁和磁盘空间加入领取前的缓存式预检。

Relay heartbeat 应上报：

- VPN 是否连接。
- NPU 主机是否可达。
- 可用芯片、Device 和 profiler。
- 当前运行任务数。
- Agent 版本和最近错误。

## 13. 阶段二：远端持久执行

本节是后续可靠性设计，当前版本尚未实现。当前 Relay 通过同步 SSH 会话执行 profiler，SSH 会话结束后再识别并回收新增结果目录。

每个任务创建固定目录：

```text
/var/lib/fla-runner/jobs/{job_id}/
  request.json
  command.json
  status.json
  result.json
  stdout.log
  stderr.log
  artifacts/
```

推荐使用：

```bash
systemd-run \
  --unit=fla-job-{job_id} \
  --working-directory=/var/lib/fla-runner/jobs/{job_id} \
  /opt/fla-runner/run-job {job_id}
```

远端执行器应原子更新 `status.json`，至少包含：

```json
{
  "job_id": "job-123",
  "state": "running",
  "pid": 12345,
  "device": 2,
  "started_at": "2026-08-08T10:00:00+08:00",
  "finished_at": null,
  "exit_code": null
}
```

Relay 重启或 VPN 恢复后，应先读取该文件和 systemd unit 状态，再决定继续跟踪、回收结果或标记异常。

## 14. Device 并发控制

当前 Worker 和 Agent 均配置 `max_concurrency=1`，可避免同一个 Relay 自身并发领取多个任务，但 NPU 服务器端尚无 Device 锁；多个 Relay 或人工任务仍可能使用同一 Device。以下为阶段二要求：

- 一个 Device 默认只运行一个采集任务。
- 锁应在 NPU 服务器端实现，不能只依赖 Relay 内存。
- 锁文件包含 `job_id`、PID、创建时间和 Agent。
- 发现旧锁时必须检查 PID/systemd unit，不能直接删除。
- 多 Agent 环境下，远端 Device 锁是最后一道互斥保证。
- 任务可根据 `chip`、`device`、`prof_tool` 和脚本能力匹配 Agent。

## 15. 命令安全

用户只提交结构化参数，命令必须由服务端白名单模板生成。

必须满足：

- `tool` 只能是允许的 `msprof`、`msprof_op`、`msprof_op_sim`。
- `script_id` 必须来自固定脚本清单。
- 数值参数有类型、上下限和组合校验。
- 文件路径不能由用户自由指定。
- kernel 名称按允许字符集和已知算子校验。
- 本地子进程使用参数数组，避免 `shell=True`。
- SSH 远端命令中的固定参数必须正确引用。
- 远端使用低权限服务账号，不使用 root 日常执行。
- SSH 使用密钥、`BatchMode=yes` 和固定 `known_hosts`。
- Runner Token、SSH 私钥和 VPN 凭据不得进入日志或浏览器。

当前部署使用 SSH 密钥、`BatchMode=yes` 和 `StrictHostKeyChecking=accept-new`，但远端账号仍为 `root`。生产化前应创建低权限专用账号、预置并固定 `known_hosts`，同时限制该账号可访问的项目目录和可执行命令。

## 16. 日志和进度

当前空闲时，Runner heartbeat 随每次轮询发送；执行任务时，独立线程每 15 秒发送 job heartbeat。当前只上报执行开始、完成、失败及连接健康状态，尚未实现以下细粒度阶段事件：

```text
checking_vpn
checking_npu
preparing_remote_dir
starting_profiler
profiling
parsing
compressing
downloading
retaining_artifacts
finalizing
```

D1 中每个任务只保留有限数量、有限长度的事件。完整 stdout/stderr 保存在本地制品目录。看板显示日志尾部，不持续把大日志写入 D1。

## 17. 结果与制品管理

### D1 保存

- 执行状态和时间。
- 退出码和失败类型。
- MFU、MBU、耗时等结构化指标。
- Agent 版本、执行模式、芯片、SoC 和 Device。
- 当前尚未采集代码 commit、CANN、profiler 和驱动版本，这些属于后续可追溯性增强。
- 本地制品标识、受控路径、大小、SHA-256 和过期时间。

### NPU 服务器或 Relay 本地保存

- 完整的 `PROF_*` / `OPPROF_*` 目录，当前不压缩。
- Prof 目录内由 CANN 生成的原始 CSV、JSON 和分析文件。
- Relay 任务状态与制品清单保存在 `data/runner-state`，Agent 运行日志保存在 `.local-secrets/runner.log`。
- 当前没有为每个任务单独持久化完整 SSH stdout/stderr，也没有自动生成可视化图片或可复现压缩包。

### 当前制品处理

- profiler 在 NPU 服务器生成目录后，Relay 通过 SCP 回收完整目录。
- Relay 使用导入器在内存中生成结构化结果，并回传 Worker/D1；Relay 模式不会改写仓库中的 `data/performance-data.json` 和 `docs/performance-data.json`。
- Relay 计算整个目录的总大小和 SHA-256，在本机生成 ZIP，并以 32 MiB 分片通过 Worker 上传到私有 R2 Bucket。
- Worker 使用 R2 multipart API 写入对象；完成后在 D1 登记 `r2://perf-artifacts/{job_id}/{artifact_id}`、ZIP 大小、SHA-256 和过期时间。
- R2 上传失败不覆盖结构化结果：任务仍可完成，D1 登记 `relay://{runner_id}/{job_id}/{directory}`，看板显示“仅 Relay 本地”。
- 用户下载时由 Worker 校验登录会话和任务归属，再代理读取私有 R2 对象；Bucket 不开放 `r2.dev` 公网访问，不向浏览器或 Relay分发 R2 API 凭据。
- 制品默认保留 30 天，由 Agent 在后续轮询中清理过期目录。
- 本地目录 `data/prof_gdr` 和 `data/prof_op` 已加入 Git 忽略列表。

### 后续带宽优化

- 在 NPU 服务器上先解析和压缩。
- 优先回传结构化摘要，不通过公网传输大文件。
- 支持按制品类型选择是否保留原始 Prof。
- 配置单任务最大本地占用量和 30 天清理策略。
- 使用 SHA-256 校验 SCP 回收结果和本地文件完整性。

## 18. 取消和重试

### 取消

1. 用户请求取消，Worker 设置 `cancel_requested=1`。
2. Relay heartbeat 收到取消状态并在本地设置取消标记。
3. 当前同步 SSH 执行不能据此终止远端 profiler；命令返回后 Relay 才回传取消结果。
4. 阶段二需通过远端持久执行器终止进程组或 systemd unit，并在确认结束后回传 `canceled`。

VPN 断开时取消请求保持待处理，不能声称已经取消。当前版本的取消不具备进程级实时性，这是已知限制。

### 重试

- 领取前失败可以自动重试。
- SSH 启动前失败可以安全重新排队。
- 远端启动后的失败默认需要状态核对。
- 每次重试生成新的 `attempt_id`，但保留原 `job_id`。
- 限制最大自动重试次数并使用退避时间。

## 19. 安全模型

- Relay 不提供公网监听端口，公网入站流量由主机防火墙默认拒绝。
- Relay 不配置任务 Webhook，不接受 Worker 主动连接。
- 用户 Token：只能提交和查询其权限范围内的任务。
- Runner Token：只能调用 `/api/runner/*`，不能管理用户和项目任务。
- Worker 中只保存 Runner Token 哈希或使用可轮换签名密钥。
- Runner Token 定期轮换并绑定 Agent ID。
- NPU 服务器不保存 Cloudflare 管理员 Token。
- 本地制品目录仅允许运行账户和受控管理员读取，不提供公网下载地址。
- Worker 对提交、领取、取消、重试和制品清单变更全部写审计日志。
- 对用户、Agent、IP 和任务类型设置速率限制。
- 任务参数、日志和制品元数据执行敏感信息过滤。

### 19.1 DDoS 和暴力请求边界

本方案不能让整个系统不存在公网攻击面，但可以避免 Relay 和 NPU 服务器直接成为公网攻击目标：

- 公网用户只访问 Cloudflare Worker，扫描和 DDoS 由 Cloudflare 边缘承接。
- Relay 只按照自身节奏发起请求，外部请求不能直接消耗 Relay 连接和本地队列。
- Worker 暂时不可用时，Relay 进入退避，任务继续保存在 D1，恢复后继续领取。
- Worker 对用户登录、任务提交和查询接口执行速率限制、配额和审计。
- Runner API 使用独立凭据、Agent 绑定、最小权限和响应体限制。
- 用户账号或 Runner Token 泄露仍可能造成恶意任务，因此必须保留命令白名单、参数校验、任务配额和 Device 并发锁。

安全目标是让公网攻击最多造成控制面延迟或拒绝服务，不能形成公网到 Relay、VPN 或 NPU 的入站连接路径。

## 20. 可观测性

建议至少监控：

- 排队任务数和最长排队时间。
- `waiting_vpn` 持续时间。
- Agent 最近 heartbeat。
- NPU 主机和 Device 可用性。
- 执行成功率、失败类型和平均时长。
- VPN 中断次数和恢复时间。
- Prof 制品大小、本地磁盘占用和清理耗时。
- `disconnected` / `orphaned` 任务数。

告警优先级：

- 高：同一 Device 疑似重复执行、结果校验失败、Runner 凭据异常。
- 中：Agent heartbeat 超时、VPN 长时间不可用、任务进入 `orphaned`。
- 低：队列增长、本地磁盘空间不足、制品接近大小限制。

## 21. 与当前项目的集成

### 可复用部分

- `backend/perf_runner.py`：SSH、SCP、命令构建和执行。
- `scripts/import_prof_gdr.py`：整网 Prof 导入。
- `scripts/import_msprof_op.py`：算子 Prof 导入。
- `backend/db.py` 中现有性能结果规范化逻辑。
- `docs/performance-dashboard.html` 中现有任务列表和指标展示。
- Cloudflare Worker 中现有登录、用户角色和审计逻辑。

### 已落地

- `backend/runner_agent.py`：Agent 注册、健康检查、自适应领取、租约 heartbeat、执行、回传和本地制品清理。
- `backend/perf_runner.py`：同步 SSH/SCP、CANN/Conda 环境准备、连接超时、新增 Prof 目录识别和结果导入。
- `scripts/import_prof_gdr.py` / `scripts/import_msprof_op.py`：支持只返回内存结果，不要求写入仓库 JSON 快照。
- `migrations/0009_add_perf_job_queue.sql`：任务、事件、结果、制品和 Agent 数据表。
- `cloudflare/worker.js`：用户任务 API、Runner API、原子领取、租约、取消、重试和结果回传。
- `docs/performance-dashboard.html`：异步提交、Runner 状态、任务状态、重试、取消和结果展示。
- `scripts/run_runner_windows.ps1`、`scripts/protect_runner_token_windows.ps1`、`scripts/install_runner_task_windows.ps1`、`scripts/start_runners_windows.ps1` 和 `scripts/stop_runners_windows.ps1`：Windows Relay 启动、DPAPI Token、计划任务安装及 A2/A5 后台任务启停。

### 待落地

- 远端任务目录、systemd unit、状态协议、进程组取消和断线 reconcile。
- NPU 服务器端 Device 锁与多 Relay 调度。
- SSH 登录、profiler、Device 和磁盘空间的领取前预检。
- 多 Relay 场景的轮询抖动和 Worker `retry_after_seconds` 支持。
- 阶段三如启用对象存储，再新增 R2 binding、上传、签名下载和生命周期策略。

## 22. 配置建议

当前 A2 Relay 的关键环境变量：

```text
RUNNER_ID=vpn-runner-windows-01
RUNNER_NAME=Windows VPN Relay A2 01
RUNNER_API_BASE=https://flash-linear-attention-npu-io-fazhenyao.fazhenyao.workers.dev
RUNNER_TOKEN=secret-from-secure-store
RUNNER_POLL_MIN_SECONDS=2
RUNNER_POLL_MAX_SECONDS=30
RUNNER_ERROR_BACKOFF_MAX_SECONDS=120
RUNNER_HEARTBEAT_SECONDS=15
RUNNER_MAX_CONCURRENCY=1

PERF_RUN_MODE=ssh
PERF_SSH_HOST=192.168.9.221
PERF_SSH_USER=root
PERF_SSH_PORT=22
PERF_SSH_IDENTITY_FILE=C:/Users/Administrator/.ssh/id_ed25519
PERF_REMOTE_WORKDIR=/workspace/fazhenyao/flash-linear-attention-npu-io
PERF_REMOTE_SCRIPT=/home/npu_user7/fazhenyao/flash-linear-attention-npu/examples/flash_gated_delta_rule.py
PERF_REMOTE_ENV_SCRIPT=/data/fazhenyao/cann/3_23/ascend-toolkit/set_env.sh
PERF_REMOTE_CONDA_SH=/data/miniconda3/etc/profile.d/conda.sh
PERF_REMOTE_CONDA_ENV=fla_dump
PERF_PROF_OUTPUT=data/prof_gdr
PERF_OP_OUTPUT=data/prof_op
PERF_NPU_DEVICE=2
PERF_CHIP=A2
PERF_SOC_VERSION=Ascend910B
```

当前 A5 Relay 的关键差异配置：

```text
RUNNER_ID=vpn-runner-windows-a5-01
RUNNER_NAME=Windows VPN Relay A5 01
RUNNER_STATE_DIR=data/runner-state/a5
RUNNER_ARTIFACT_ROOTS=data/runner-artifacts/a5/prof_gdr;data/runner-artifacts/a5/prof_op

PERF_SSH_HOST=192.168.13.241
PERF_SSH_USER=npu_user7
PERF_SSH_PORT=22
PERF_SSH_IDENTITY_FILE=C:/Users/Administrator/.ssh/id_rsa
PERF_REMOTE_WORKDIR=/home/npu_user7/fazhenyao/flash-linear-attention-npu-io
PERF_REMOTE_SCRIPT=scripts/flash_gated_delta_rule.py
PERF_REMOTE_ENV_SCRIPT=/home/npu_user7/fazhenyao/cann/7_20/ascend-toolkit/set_env.sh
PERF_REMOTE_CONDA_SH=/home/npu_user7/jianshuqiang/miniconda3/etc/profile.d/conda.sh
PERF_REMOTE_CONDA_ENV=f30077529
PERF_PROF_OUTPUT=data/prof_gdr
PERF_OP_OUTPUT=data/prof_op
PERF_LOCAL_PROF_OUTPUT=data/runner-artifacts/a5/prof_gdr
PERF_LOCAL_OP_OUTPUT=data/runner-artifacts/a5/prof_op
PERF_NPU_DEVICE=7
PERF_CHIP=A5
PERF_SOC_VERSION=Ascend950PR
```

Windows Relay 将 A2/A5 非口令配置分别保存在 Git 忽略的 `.local-secrets/runner-config.json` 和 `.local-secrets/runner-config-a5.json`，共享的 `RUNNER_TOKEN` 通过 DPAPI 加密保存在 `.local-secrets/runner-token.clixml`。Token 和 SSH 私钥不得提交到 Git。A2 当前使用 `root` 账号仅是过渡配置，生产化前应改为低权限专用账号。

## 23. 分阶段实施

### 阶段一：最小闭环（已完成）

- 新增 `perf_jobs`、`perf_job_events` 和 `runner_agents`。
- Worker 支持提交、查询、领取、heartbeat、完成和失败。
- 在 VPN 本机运行单实例 Relay。
- Relay 复用现有 SSH 执行逻辑。
- 结果先以结构化 JSON 回传，原始文件暂存在 NPU 服务器或 Relay 本地磁盘。
- D1 只保存指标、状态、制品元数据和截断日志，不保存 Prof 压缩包或完整 stdout/stderr。
- 本地制品默认保留 30 天，由定时清理任务删除过期目录。

验收条件：VPN 上线后任务可自动执行；VPN 离线时任务可靠等待；看板能展示最终指标。

2026-08-08 已在 A2 NPU 2 上完成两条真实 `msprof` 端到端任务：`T=128` 总耗时 `1.0 ms`，`T=4087` 总耗时 `15.2 ms`。同日又在 A5 NPU 7 上完成 `T=128` 的定向 `msprof` 任务，总耗时 `0.7 ms`；切换至 `/home/npu_user7/fazhenyao/flash-linear-attention-npu/examples/flash_gated_delta_rule.py` 后再次执行同一用例，结果仍为 `0.7 ms`，并成功回收约 `1.66 MB` 的 PROF 目录。这些任务均由看板提交、Worker/D1 排队、指定 Windows Relay 领取、SSH 到 NPU 执行、SCP 回收，并最终在看板显示“已完成”。

### 阶段二：可靠性

- 远端改用 systemd 持久执行。
- 增加断线恢复、取消、重试和 Device 锁。
- 引入 `attempt_id`、租约和幂等键。
- 增加 Agent 健康状态和告警。

验收条件：执行中断开 VPN 后，恢复连接可以继续追踪同一任务且不会重复执行。

### 阶段三：云端制品管理

- 使用私有 R2 Bucket 保存压缩 Prof。
- Relay 只使用已有 Runner Token，经 Worker 执行 multipart 上传，不保存 R2 API Token。
- Worker 校验用户权限并代理下载，D1 只保存制品元数据。
- 使用 SHA-256、30 天过期时间和 Bucket 生命周期策略控制完整性与容量。

验收条件：用户可以下载受控制品，D1 不存储大文件。

### 阶段四：生产化

- 迁移到专用 VPN Relay 或站点到站点网络。
- 当前已支持多个 Agent、A2/A5 芯片和指定 Device 的精确路由；后续增加容量感知和自动调度。
- 增加权限细分、速率限制、审计和运行指标。
- 建立自动化 API、状态机、断线恢复和权限测试。

## 24. 验收测试场景

1. VPN 未连接时提交任务，任务保持等待且不失败。
2. VPN 恢复后 Relay 自动领取并执行任务。
3. 两个 Relay 同时领取时只有一个成功。
4. 同一 Device 同时提交两个任务时串行执行。
5. SSH 启动前断开 VPN，任务可以安全重新排队。
6. 远端运行中断开 VPN，命令继续执行且不会重复调度。
7. VPN 恢复后 Relay 找回远端任务并回传结果。
8. Relay 重启后根据 `job_id` 恢复任务。
9. 用户取消任务后远端进程和子进程全部结束。
10. 重复点击提交不会创建重复任务。
11. 非白名单脚本、非法参数和 Shell 字符被拒绝。
12. 大文件上传失败时结构化结果仍保留，并可单独重试上传。
13. 制品 SHA-256 不一致时任务不能标记为完整成功。
14. Runner Token 不能访问管理员接口。
15. 普通用户不能查看或取消无权限的任务。

## 25. 当前部署决策

当前采用“Worker/D1 队列 + VPN Runner Relay 主动拉取 + 同步 SSH 远端执行 + Relay 本地副本 + 私有 R2 在线下载”。

- Worker/D1 保存任务状态、结构化性能指标、审计信息和制品清单。
- 看板允许显式选择 A2 或 A5 执行服务器，Worker 使用 `target_runner_id` 保证任务只被指定 Relay 领取。
- 两个 Relay Agent 运行在同一台 Windows VPN 电脑上，各自连接一台 NPU 服务器，配置、状态、日志和本地制品目录互相隔离。
- Relay 保存回收后的原始 Prof 和分析文件，默认保留 30 天；同时压缩并分片上传到 R2，NPU 服务器源目录当前需要单独清理。
- 登录用户可从任意电脑通过看板下载自己提交任务的 R2 制品，管理员可下载全部任务制品。
- Relay 仍只发起出站 HTTPS 请求，不开放公网入站端口。
- R2 上传失败不影响结构化指标回传；原始制品退化为 Relay 本地保存。

不允许把大型 Prof 压缩包、完整 stdout/stderr 或图片写入 D1。R2 Bucket 保持私有，下载必须经过 Worker 的任务权限校验；Relay 和浏览器均不持有 R2 凭据。

Relay 即使具备公网入站和出站能力，也不作为公网服务使用：

- 不部署公网 Webhook。
- 不运行 Cloudflare Tunnel。
- 不开放任务接收端口。
- 不接受 Worker 主动连接。
- 主机防火墙拒绝公网入站。
- 通过自适应出站 HTTPS 请求领取任务并回传结果。

该决策优先保证 Relay 和 VPN 内网不暴露给公网，接受空队列请求和任务启动延迟作为交换。Relay 长时间空闲时轮询间隔达到 30 秒，因此当前典型领取延迟为 `0～30 秒`。

### 25.1 未选择：公网 Webhook

公网 Webhook 可以降低通知延迟和空请求数量，但会让 Relay 成为公网服务，需要额外承担域名、TLS、WAF、身份认证、限流、DDoS、漏洞修复和公网到 VPN 边界隔离责任。该风险不满足当前安全要求，因此不采用。

### 25.2 未选择：Cloudflare Tunnel Webhook

Tunnel 可以避免直接开放源站端口，但逻辑上仍提供一个可被 Cloudflare 访问的 Relay 入站应用，并引入 Tunnel、Access、Webhook 重试和本地持久通知队列。当前没有必要为较低通知延迟承担这部分复杂度，因此不采用。

### 25.3 可选演进：出站持久连接

如果未来对秒级通知有明确需求，可以让 Relay 主动建立到 Cloudflare Durable Object 或受控消息代理的出站 WebSocket，由云端沿已有连接发送通知。该方案不要求 Relay 开放公网端口，但必须实现：

- 自动重连和心跳。
- 消息序号与游标。
- 重连后的漏消息对账。
- D1 任务事实来源和原子接受。
- 连接服务的鉴权、容量和故障恢复。

在这些能力实现前，自适应 HTTPS 领取是默认方案。

第一阶段允许 Relay 运行在当前开启 VPN 的本机，以最小改造复用现有 `perf_runner.py`。生产阶段应迁移到长期在线的专用 VPN 网关或获得受控的站点到站点网络能力。

该架构将公网控制面和 NPU 执行面分离：Cloudflare 负责可靠传递、认证和记录任务，VPN Relay 负责网络边界，NPU 服务器只负责受控执行。

## 26. 实施状态

截至 2026-08-10，阶段一闭环已经部署并通过真实 NPU 任务验证，阶段三 R2 代码已完成：

- D1 migration 已加入 `perf_jobs`、`perf_job_events`、`perf_results`、`perf_artifacts` 和 `runner_agents`。
- Worker 已实现用户任务 API、Runner API、幂等提交、原子领取、租约、heartbeat、取消、重试和结果/制品清单回传。
- `backend/runner_agent.py` 已实现主动出站注册、SSH 端口健康检查、自适应领取、执行期 heartbeat、结果回传和本地制品过期清理。
- `backend/perf_runner.py` 已支持远端 CANN/Conda 环境准备、同步 SSH 执行、SSH/SCP 超时、新增结果目录识别和回收。
- 性能看板已改为向 Worker 异步提交任务，并提供 A2/A5 执行服务器选择；Runner 或 VPN 离线时任务继续排队。
- 已增加不执行 NPU 命令的队列冒烟测试 `scripts/smoke_test_perf_queue.py`。
- Worker 和 Relay 已配置独立 `RUNNER_TOKEN`，本机 Token 使用 DPAPI 加密保存。
- Windows 计划任务 `FLA VPN Runner` 和 `FLA VPN Runner A5` 已安装并运行；两个计划任务都在用户登录后启动 Agent，以复用本机 VPN 会话。
- 已验证看板同时显示两个在线 Runner，并完成 A2 NPU 2 与 A5 NPU 7 的真实 `msprof` 任务领取、执行、回收、解析和结果展示。
- Relay 导入结果时只构建内存数据并回传 D1，不再改写仓库性能快照；本地 Prof 目录已加入 Git 忽略列表。
- A5 的芯片频率和 HBM 带宽理论常量尚未录入，A5 结果中的 MFU/MBU 保持空值，不使用 A2 常量代算。
- Worker 已增加私有 R2 binding、multipart 上传、任务归属鉴权下载和制品元数据接口；Relay 已增加 ZIP、分片上传及失败降级逻辑；看板已增加云端制品下载入口。
- Cloudflare R2 Bucket `flash-linear-attention-npu-perf-artifacts` 已创建，默认存储类为 Standard，公开访问已禁用。

### 26.1 当前 Windows Relay 的本机凭据与启动方式

当前过渡阶段 Relay 运行在用户开启 VPN 的 Windows 电脑上，采用以下本机约束：

- `RUNNER_TOKEN` 使用 Windows DPAPI 加密到 `.local-secrets/runner-token.clixml`，只允许创建它的 Windows 用户解密。
- NPU 地址、SSH 用户、SSH 私钥路径等非口令配置分别保存在 `.local-secrets/runner-config.json` 和 `.local-secrets/runner-config-a5.json`，该目录已被 Git 忽略。
- Agent 通过 Windows 计划任务在当前用户登录后启动，以便复用交互式 VPN 会话；VPN 未连接时 Agent 不领取任务，VPN 恢复后自动继续。
- 计划任务通过 GUI 子系统的 `wscript.exe` 运行 `scripts/run_runner_windows_hidden.vbs`，再以无控制台方式启动 `scripts/run_runner_windows.ps1` 和 Python Agent。Relay 与启动它的终端完全分离，不注册 Windows 入站服务、不配置公网域名、不开放本机端口。
- 管理员使用 `scripts/start_runners_windows.ps1` 和 `scripts/stop_runners_windows.ps1` 同时启动或停止 A2/A5，也可通过 `-TaskName` 只控制指定计划任务。
- A2/A5 Relay 日志分别写入 `.local-secrets/runner.log` 和 `.local-secrets/runner-a5.log`；状态分别写入 `data/runner-state` 和 `data/runner-state/a5`。
- A2 原始性能制品保存在既有 `data/prof_gdr` / `data/prof_op`，A5 制品保存在 `data/runner-artifacts/a5/prof_gdr` / `data/runner-artifacts/a5/prof_op`，并执行 30 天保留期清理。
- 两个 Agent 均为单并发。A2 默认 NPU 2、SoC `Ascend910B`；A5 默认 NPU 7、SoC `Ascend950PR`。

### 26.2 当前已知限制

- 同步 SSH 执行依赖连接持续存在；执行期 VPN/SSH 中断后尚不能通过远端状态文件自动恢复。
- 取消请求只能设置本地标记，尚不能实时终止远端 profiler 进程。
- 服务器端尚未实现 Device 锁，多 Relay 或人工任务需要运维协调。
- 领取前只检查 SSH TCP 端口，SSH 认证、CANN/Conda、profiler、Device 和磁盘空间在执行阶段才暴露错误。
- A2 当前使用 `root` SSH 账号；A2/A5 均使用 `StrictHostKeyChecking=accept-new`。A2 应迁移到低权限账号，两台服务器都应预置固定主机密钥。
- 当前没有轮询随机抖动，也不读取 Worker 返回的建议重试间隔；双 Relay 已投入运行，应尽快补齐抖动和服务端退避提示支持。
- 30 天过期清理只覆盖 Relay 本地副本，不会删除 NPU 服务器 `data/prof_gdr` / `data/prof_op` 下的源目录。
- R2 对象和 Relay 本地副本都按 30 天目标保留；D1 过期元数据目前不会由定时任务主动删除。
- Windows 电脑、登录会话或 VPN 离线时，任务会继续保存在 D1 中，但不会执行。

### 26.3 下一阶段优先级

1. 建立低权限 NPU 服务账号和固定 `known_hosts`。
2. 实现远端任务目录、进程组/systemd 持久执行、Device 锁和状态文件。
3. 实现执行期取消、Agent 重启恢复和 Worker reconcile。
4. 增加领取前 profiler/Device/磁盘预检、轮询抖动和服务端退避提示支持。
5. 增加 R2 上传失败的独立重试、过期 D1 元数据清理和制品下载审计。

## 27. R2 部署配置

- Bucket 名称：`flash-linear-attention-npu-perf-artifacts`。
- Worker binding：`PERF_ARTIFACTS`，配置在 `wrangler.toml`。
- 存储类型：Standard；Bucket 保持私有，不启用 `r2.dev` 或自定义公开域名。
- 生命周期：对前缀 `perf-artifacts/` 设置 30 天删除规则；未完成 multipart upload 由 R2 自动清理。
- Relay 分片：`RUNNER_R2_PART_MIB=32`，不得低于 5 MiB。
- 下载授权：普通用户只能访问本人创建的任务，管理员可访问全部任务；Worker 返回 `private, no-store`。
- 完整性：Relay 对最终 ZIP 计算 SHA-256，D1 保存摘要，下载响应返回 `X-Content-SHA256`。
