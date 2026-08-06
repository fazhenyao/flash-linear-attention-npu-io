// Cloudflare Worker API base URL.
window.FLASH_IO_API_BASE = "https://flash-linear-attention-npu-io.polludvtur.workers.dev";
// 本地看板服务将 /api/* 同源代理到真实 NPU 后端。
// The dashboard server proxies /api/* to the local NPU backend. Using the
// current origin also works when the dashboard is opened from another user's
// browser; their browser must not try to call its own 127.0.0.1.
window.FLASH_IO_LOCAL_API = window.location.origin;
