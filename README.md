# flash-linear-attention-npu IO 控制台

这是一个私有项目管理控制台骨架，用于维护 `flash-linear-attention-npu` 转测计划。

## 能力

- 前端：原生 HTML/CSS/JavaScript，入口 `/io`
- 后端：Python 标准库 HTTP API
- 后端数据库：SQLite `data/project.sqlite3`，作为任务、分组、专项和审计日志的主数据源
- 公网后端：Cloudflare Workers + D1，可作为低成本公网 API 和 SQL 数据库
- 数据快照：`data/project-state.json`，由 SQLite 导出，用于备份和 GitHub Pages 只读展示
- 审计快照：`data/audit-log.jsonl`，由 SQLite 审计表导出，便于 grep/脚本检索
- 数据操作：任务增删改查、筛选、分组、专项、甘特图展示、双击甘特条切分、JSON 导出
- 每次写操作都会先写入 SQLite，再导出快照，并自动提交推送到 GitHub 私有仓库

## 本地运行

```powershell
python .\backend\app.py
```

浏览器打开：

```text
http://127.0.0.1:8787/io
```

首次启动如果 SQLite 为空，会从 `data/project-state.json` 和 `data/audit-log.jsonl` 初始化；如果快照不存在，则从内置 seed 数据初始化。后续以 SQLite 为准，不会在每次启动时用 JSON 覆盖数据库。

## GitHub Pages 查看

本仓库使用 `docs/` 作为 GitHub Pages 静态站点目录。

- 本地 `/io` 页面通过后端 API 编辑 SQLite 数据库。
- 每次本地写操作会从 SQLite 导出 `data/project-state.json` 和 `data/audit-log.jsonl`。
- 同时会把最新数据镜像到 `docs/project-state.json` 和 `docs/audit-log.jsonl`。
- GitHub Pages 页面只读展示仓库里的最新数据，不运行 Python 后端。
- GitHub Pages 页面也支持编辑模式：输入你自己的 GitHub fine-grained token 后，可以直接在网页上增删改查并写回仓库。

预期访问地址：

```text
https://fazhenyao.github.io/flash-linear-attention-npu-io/
```

## Cloudflare Workers + D1 公网部署

Cloudflare 方案用于把 GitHub Pages 前端接到公网后端数据库。部署后：

- 前端仍可放在 GitHub Pages。
- Worker 提供 `/api/export`、`/api/version`、`/api/tasks/:id`、`/api/groups/:id`、`/api/specials/:id`、`/api/people/:id`、`/api/save`、`/api/audit`、`/api/pr-catalog`。
- D1 存任务、人员、分组、专项、甘特分段和审计日志。
- `docs/config.js` 配置 Worker URL 后，页面读写都会走 Cloudflare D1。

准备 Node.js 后安装依赖：

```powershell
npm install
```

登录 Cloudflare：

```powershell
npx wrangler login
```

创建 D1 数据库：

```powershell
npx wrangler d1 create flash-linear-attention-npu-io-fazhenyao
```

把命令输出里的 `database_id` 填入 `wrangler.toml` 的 `database_id`。

设置写入密钥：

```powershell
npx wrangler secret put ADMIN_TOKEN
npx wrangler secret put AUTH_SECRET
```

应用 D1 表结构：

```powershell
npm run cf:migrate:remote
```

部署 Worker：

```powershell
npm run cf:deploy
```

把现有仓库快照导入 D1：

```powershell
python .\scripts\import_cloudflare.py --api https://你的-worker-url --token 你的-ADMIN_TOKEN
```

创建管理员账号：

```powershell
python .\scripts\create_cloudflare_user.py --api https://你的-worker-url --admin-token 你的-ADMIN_TOKEN --username admin --password 一个强密码 --role admin --display-name 管理员 --owner-name 管理员
```

同名账号已存在时，该脚本会更新该账号的密码、角色、显示名和负责人，不会改动项目任务数据。

创建开发账号时，`--owner-name` 要写成任务里的责任人姓名。开发账号默认只能更新自己名下的既有任务：

```powershell
python .\scripts\create_cloudflare_user.py --api https://你的-worker-url --admin-token 你的-ADMIN_TOKEN --username dev-chen --password 一个强密码 --role developer --display-name 陈琳鑫 --owner-name 陈琳鑫
```

配置 GitHub Pages 前端使用 Worker：

```javascript
// docs/config.js
window.FLASH_IO_API_BASE = "https://你的-worker-url";
```

提交并推送 `docs/config.js` 后，公开页面会从 Cloudflare D1 读取数据。编辑模式下使用账号密码登录，不再输入 GitHub token。日常编辑走字段级接口：任务变更走 `PATCH /api/tasks/:id`，新增/删除任务走 `POST/DELETE /api/tasks`，分组、专项、人员分别走自己的 `POST/PATCH/DELETE` 接口。`POST /api/save` 仅保留为导入、兼容和应急兜底入口。

### GitHub 到 Cloudflare 的快速部署链路

本仓库已配置 `.github/workflows/deploy-cloudflare.yml`：

- 修改 `cloudflare/**`、`migrations/**`、`wrangler.toml` 或部署 workflow 后，push 到 `main` 会自动部署 Worker。
- 自动部署会先校验 `wrangler.toml`，再执行远端 D1 migration，最后执行 `wrangler deploy`。
- 也可以在 GitHub Actions 页面手动运行 `Deploy Cloudflare Worker`，并选择是否执行 migration、是否部署 Worker。

GitHub 仓库需要配置以下 Actions Secrets：

- `CLOUDFLARE_ACCOUNT_ID`：Cloudflare 账号 ID。
- `CLOUDFLARE_API_TOKEN`：Cloudflare API Token，不要写入仓库。
- `FLASH_IO_ADMIN_TOKEN`：Worker 的 `ADMIN_TOKEN` 值，仅用于定时 PR 候选池同步和 D1 快照备份，不要写入仓库。
- `CLOUDFLARE_AUTH_SECRET`：Worker 登录令牌签名密钥，仅在首次部署或主动轮换时写入 Worker。

首次手动运行 `Deploy Cloudflare Worker` 时启用 `configure_secrets` 和 `import_snapshots`。初始化导入只覆盖任务、人员、算子、审计、PR 目录和性能快照，不复制旧用户、密码哈希或登录会话。

### 人员账号初始化与改密

可在 GitHub Actions 页面手动运行 `Initialize Cloudflare Users`，为 D1 人员表中的每个人生成开发账号：

- 登录账号默认使用人员姓名。
- 跳过 `待填写`、`待排人力`、`对应算子责任人` 等占位人员。
- 初始密码写入 artifact `initial-person-accounts`，保留 7 天，不提交到仓库。
- `reset_existing=false` 时只补齐缺失账号；`reset_existing=true` 时会重置已有人员账号密码。

本地也可以直接运行：

```powershell
python .\scripts\bootstrap_cloudflare_users.py --api https://你的-worker-url --token 你的-ADMIN_TOKEN --out .\generated\initial-accounts.csv --reset-existing
```

用户登录后可以点击“修改密码”。页面会要求输入旧密码、新密码、再次确认新密码，并在最终确认弹窗确认后才提交到后端生效。

### 页面实时同步

前端会每 3 秒查询 `/api/version`：

- 如果后台数据版本变化，且当前页面没有未保存修改，会自动重新加载 D1 数据。
- 如果当前页面有未保存修改，不会自动覆盖本地编辑，只会提示“后台已有新数据”。
- 这个同步是近实时轮询，不需要额外部署 WebSocket 或 Durable Objects。

日常修改路径：

1. 改 Worker API 或权限逻辑：修改 `cloudflare/worker.js`，提交并 push，Actions 自动发布。
2. 改数据库结构：新增 `migrations/0002_xxx.sql`，提交并 push，Actions 自动应用 migration 并发布。
3. 改前端页面：修改 `docs/**`，GitHub Pages 按仓库 Pages 设置发布；不需要 Worker 部署。
4. 改 Worker 地址：修改 `docs/config.js`，提交并 push，GitHub Pages 生效后前端切换到新 API。

快速回滚：

1. Worker 代码回滚：`git revert <有问题的提交>` 后 push，Actions 会自动重新部署旧逻辑。
2. 前端回滚：`git revert <有问题的提交>` 后 push，GitHub Pages 自动更新。
3. 数据库结构回滚：不要直接改旧 migration；新增一个反向 migration。高风险 migration 前建议先用 Cloudflare D1 导出备份。

### D1 备份与恢复

本仓库配置 `.github/workflows/backup-d1.yml`，每天北京时间 02:00 自动备份，也可以在 GitHub Actions 页面手动运行 `Backup Cloudflare D1`。

备份内容：

- `data/project-state.json` 和 `docs/project-state.json`：D1 当前项目状态快照。
- `data/audit-log.jsonl` 和 `docs/audit-log.jsonl`：完整审计日志快照。
- `data/pr-catalog.json` 和 `docs/pr-catalog.json`：PR 候选池快照。
- `data/performance-data.json` 和 `docs/performance-data.json`：结构化性能指标和运行快照。
- Actions artifact `d1-latest-sql`：完整 D1 SQL 导出，保留 30 天；该文件可能包含账号密码哈希和盐，只作为私有恢复材料，不提交到仓库。

优先恢复方式是 D1 Time Travel：

```powershell
npx wrangler d1 time-travel info flash-linear-attention-npu-io-fazhenyao --timestamp="2026-06-19T10:00:00+08:00"
npx wrangler d1 time-travel restore flash-linear-attention-npu-io-fazhenyao --timestamp="2026-06-19T10:00:00+08:00"
```

恢复前建议先导出事故现场：

```powershell
npx wrangler d1 export flash-linear-attention-npu-io-fazhenyao --remote --output=./backup-before-restore.sql
```

如果 D1 被误删或不可用，可新建 D1、应用 migrations，再从仓库 JSON 快照恢复业务数据：

```powershell
npx wrangler d1 create flash-linear-attention-npu-io-fazhenyao
npm run cf:migrate:remote
python .\scripts\import_cloudflare.py --api https://你的-worker-url --token 你的-ADMIN_TOKEN
```

如果需要完整恢复账号表等整库内容，可下载最近一次 `d1-latest-sql` artifact 后导入：

```powershell
npx wrangler d1 execute flash-linear-attention-npu-io-fazhenyao --remote --file=.\d1-latest.sql
```

本地快速验证：

```powershell
python -m py_compile scripts\import_cloudflare.py scripts\create_cloudflare_user.py scripts\backup_cloudflare.py
python -c "import sqlite3,pathlib; conn=sqlite3.connect(':memory:'); conn.executescript(pathlib.Path('migrations/0001_init.sql').read_text(encoding='utf-8'))"
node --check cloudflare/worker.js
```

## 仓库数据文件

- `data/project.sqlite3`：后端数据库，包含任务、分组、专项、甘特分段和审计日志，已被 `.gitignore` 忽略。
- `data/project-state.json`：从数据库导出的最新项目数据快照。
- `data/audit-log.jsonl`：从数据库导出的追加式操作日志，每行是一条 JSON 记录。

如果 GitHub 访问需要代理，服务会优先读取环境变量 `HTTP_PROXY` / `HTTPS_PROXY`；未设置时会尝试读取 Windows 用户代理配置。

## Pages 编辑模式

公开页面默认只读。

- 未配置 `docs/config.js` 的 Worker URL 时，点击“启用编辑”，输入 GitHub fine-grained token 写回仓库文件。
- 配置 Worker URL 后，使用账号密码登录，写回 Cloudflare D1。

Cloudflare 权限模型：

- `admin`：可全量修改任务、人员、分组、专项和项目数据。
- `developer`：进入编辑模式后，仅展示自己负责的任务，以及与自己任务属于同一算子的关联任务；只能给责任人字段包含自己 `ownerName` 的既有任务提交 `PR 链接` 和 `转测报告`，同算子关联任务只读；不能新增/删除任务、人员、分组和专项，也不能调整排期、风险、优先级或状态。
- 密码不会明文存储，Worker 会保存加盐 PBKDF2-SHA256 哈希。

建议 token 设置：

- Repository access：只选择 `fazhenyao/flash-linear-attention-npu-io`
- Permissions：`Contents: Read and write`
- Expiration：尽量短，例如 7 天

安全规则：

- 不要把 token 写入仓库。
- 页面只把 token 保存在当前浏览器会话的 `sessionStorage`。
- 退出编辑或关闭浏览器会话后需要重新输入。
- 可以单行保存，也可以连续修改多行后点击“保存全部”，一次 GitHub commit 写回所有待保存任务。
- 每次写回都会更新 `data/` 和 `docs/` 下的数据快照和审计日志。

## API

Cloudflare Worker API：

- `GET /api/health`
- `GET /api/export`
- `GET /api/state`
- `GET /api/version`
- `GET /api/audit`
- `GET /api/audit/export`
- `GET /api/pr-catalog`
- `POST /api/tasks`
- `PATCH /api/tasks/{id}`
- `DELETE /api/tasks/{id}`
- `POST /api/groups`
- `PATCH /api/groups/{id}`
- `DELETE /api/groups/{id}`
- `POST /api/specials`
- `PATCH /api/specials/{id}`
- `DELETE /api/specials/{id}`
- `POST /api/people`
- `PATCH /api/people/{id}`
- `DELETE /api/people/{id}`
- `POST /api/me/password`
- `GET /api/users`
- `POST /api/users`
- `POST /api/save`
- `POST /api/import`

本地 Python API：

- `GET /api/health`
- `GET /api/audit`
- `GET /api/tasks`
- `POST /api/tasks`
- `PATCH /api/tasks/{id}`
- `DELETE /api/tasks/{id}`
- `POST /api/tasks/{id}/split`
- `GET /api/groups`
- `POST /api/groups`
- `PATCH /api/groups/{id}`
- `DELETE /api/groups/{id}`
- `GET /api/specials`
- `POST /api/specials`
- `PATCH /api/specials/{id}`
- `DELETE /api/specials/{id}`
- `GET /api/export`

## 同步说明

本地服务写入数据后会自动执行 `git commit` 和 `git push`。如果网络需要代理，请先保证 Windows 用户代理或 `HTTP_PROXY` / `HTTPS_PROXY` 可用。
