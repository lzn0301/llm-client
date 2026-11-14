# llm-client

面向多项目复用的轻量化 Python LLM 公共客户端库，基于官方 `google-genai` 与 `httpx` 构建，聚焦 **稳定性防卡死**、**逻辑档位解耦模型换代** 与 **跨厂商自动兜底**。

## 核心特性

- **逻辑档位路由与动态注册表**：通过 `lite` / `flash` / `pro` / `image` 等档位解耦具体模型名；支持本地 `model_registry.json` 与远端 `LLM_REGISTRY_URL` 热更新（带内存缓存与内置默认表兜底）。
- **强制单次请求超时**：为每次 Gemini 调用显式注入 `HttpOptions(timeout=ms)`，并使用 `httpx` 客户端超时控制，避免连接假死或无限等待。
- **自动重试与同厂模型降级**：遇到 `503`、`429`、`500`、超时、`Server disconnected` 或 `404`（旧模型下线）时自动按优先级切换至下一备选模型；当目标模型不支持 `thinking_config` 时自动剥离后重试。
- **跨厂商无缝兜底**：当 Gemini 队列全线不可用且配置了 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 时，纯文本/JSON 请求自动切换至 DeepSeek / OpenAI 兼容接口继续完成任务。
- **全模态能力覆盖**：统一提供 `generate()`、`generate_json()`、`generate_image()` / `generate_images()`、`create_chat()`、`count_tokens()`、`upload_file()` / `delete_file()`，并支持 `usage_callback` 成本统计钩子。

## 安装

本地开发可编辑安装：

```bash
pip install -e ../llm-client
```

或从 GitHub 安装：

```bash
pip install git+https://github.com/lzn0301/llm-client.git
```

## 环境变量

- `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`：Google Gemini API 密钥
- `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`：跨厂兜底（DeepSeek / OpenAI 兼容）密钥
- `LLM_FALLBACK_BASE_URL` 或 `OPENAI_BASE_URL`：可选，兼容接口基础地址（使用 DeepSeek 时默认 `https://api.deepseek.com/v1`）
- `LLM_REGISTRY_URL`：可选，远端 `model_registry.json` 拉取地址

## 使用示例

```python
from llm_client import LLMClient

client = LLMClient()

# 1. 按逻辑档位调用文本生成（内置超时 + 自动降级 + DeepSeek 兜底）
text = client.generate("用一句话总结今天的重点", tier="flash")

# 2. 结构化 JSON 提取
data = client.generate_json("返回包含 title 和 tags 的 JSON 对象", tier="lite")

# 3. 原生图片生成
image_bytes = client.generate_image("A serene emerald pendant on white linen", tier="image")
```
