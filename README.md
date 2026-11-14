# llm-client

面向多项目复用的轻量化 Python LLM 公共客户端库，基于官方 `google-genai` 与 `httpx` 构建，聚焦 **集中统一密钥管理**、**免费/付费分层路由**、**稳定性防卡死** 与 **跨厂商自动兜底**。

## 核心特性

- **本机全局密钥统一管理**：自动从 `~/.config/llm-client/.env`（Windows 为 `C:\Users\<用户名>\.config\llm-client\.env`）加载共享 API Key，外层业务项目无需在每个 `.env` 中重复复制密钥；同时严格保留「显式参数 > 进程/项目 `.env` > 全局 `.env`」的覆盖优先级，兼容 GitHub Actions / Docker 云端部署。
- **免费池 / 付费池分层路由 (`LLM_ROUTING_MODE`)**：支持 `free_first`（免费 Key 优先，限流/配额耗尽自动切付费 Key）、`paid_first`（付费 Key 优先，免费池备援）、`free_only`、`paid_only` 四种模式，可按项目单独配置。
- **占位符密钥过滤**：自动识别并忽略 `.env` 模板中遗留的 `your-openai-api-key`、`xxxxx`、`<API_KEY>` 等占位符，避免无效兜底请求。
- **逻辑档位路由与动态注册表**：通过 `lite` / `flash` / `pro` / `image` 等档位解耦具体模型名（`flash` 默认优先 `gemini-4-hydrogen`）；支持本地 `model_registry.json` 与远端 `LLM_REGISTRY_URL` 热更新。
- **强制单次请求超时与自动重试**：为每次 Gemini 调用显式注入 `HttpOptions(timeout=ms)`，遇到 `503`、`429`、`500`、超时、`Server disconnected` 或 `404` 自动降级切换模型与密钥层级。
- **跨厂商无缝兜底**：当 Gemini 所有候选模型与密钥层级均不可用且配置了 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 时，纯文本/JSON 请求自动切换至 DeepSeek / OpenAI 兼容接口继续完成任务。

## 安装

本地开发可编辑安装：

```bash
pip install -e ../llm-client
```

或从 GitHub 安装：

```bash
pip install git+https://github.com/lzn0301/llm-client.git@main
```

## 环境变量配置

推荐在 `~/.config/llm-client/.env` 中集中存放密钥（切勿提交至任何 Git 仓库）：

```env
GEMINI_FREE_API_KEYS=你的免费GeminiKey1,可选免费Key2
GEMINI_PAID_API_KEY=你的付费GeminiKey
DEEPSEEK_API_KEY=你的DeepSeekKey
LLM_ROUTING_MODE=free_first
```

外层项目自身 `.env` 中只需按业务需要配置路由策略（无需再写 Key）：

```env
LLM_ROUTING_MODE=paid_first
```

支持的变量说明：
- `GEMINI_FREE_API_KEYS`：免费 Gemini API Key 列表（英文逗号分隔）
- `GEMINI_PAID_API_KEY` / `GEMINI_PAID_API_KEYS`：付费 Gemini API Key
- `GEMINI_API_KEY` / `GOOGLE_API_KEY`：单 Key 模式（如 GitHub Actions CI 或单项目覆盖）
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`：跨厂商兜底密钥
- `LLM_ROUTING_MODE`：可选 `free_first`（默认）、`paid_first`、`free_only`、`paid_only`
- `LLM_FALLBACK_BASE_URL` / `OPENAI_BASE_URL`：可选，兼容接口基础地址（使用 DeepSeek 时默认 `https://api.deepseek.com/v1`）
- `LLM_CLIENT_ENV_FILE`：可选，指定全局配置文件路径（设为 `""` 可禁用读取全局文件）
- `LLM_REGISTRY_URL`：可选，远端 `model_registry.json` 拉取地址

## 使用示例

```python
from llm_client import LLMClient

client = LLMClient()

# 1. 按逻辑档位调用文本生成（自动读取全局 ~/.config/llm-client/.env 并按路由策略执行降级）
text = client.generate("用一句话总结今天的重点", tier="flash")

# 2. 显式指定项目路由策略
paid_client = LLMClient(routing_mode="paid_first")
data = paid_client.generate_json("返回包含 title 和 tags 的 JSON 对象", tier="flash")

# 3. 原生图片生成（若免费 Key 不支持或限流，在 free_first 下也会自动切到付费 Key）
image_bytes = client.generate_image("A serene emerald pendant on white linen", tier="image")
```
