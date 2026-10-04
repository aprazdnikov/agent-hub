# Бэкенд Codex

## Цель

OpenAI Codex — второй агент рядом с Claude, выбираемый на уровне темы, с авторизацией по API-ключу или по подписке ChatGPT. Паритет с Claude: текст, вызовы инструментов, одобрения кнопками, отправка файлов, фото на вход, досылка сообщений посреди хода, вопросы кнопками.

## Решения

| Вопрос | Решение |
|---|---|
| Паритет | Полный, кроме фоновых задач (у Codex их нет) |
| Интеграция | Собственный asyncio JSON-RPC клиент к `codex app-server` (stdio). Бинарник — пакет `openai-codex-cli-bin` (колёса для Linux/macOS/Windows) |
| Авторизация | Подписка ChatGPT: `codex login --device-auth`, данные в `$CODEX_HOME/auth.json`. API-ключ: `OPENAI_API_KEY` в окружении хаба; при старте хаб передаёт его в Codex через `account/login/start` (app-server сам переменные окружения не читает), ключ сохраняется в `$CODEX_HOME`. Ключ применяется, только если входа нет или это уже вход по ключу (так обновляется сменённый ключ); вход по подписке ChatGPT сохраняется — для перехода на ключ `codex logout`. При старте `account/read`: нет входа → `warning` в логе, хаб работает дальше |
| Выбор агента | `/new codex <path>`, `AGENT_HUB_DEFAULT_BACKEND`, `/backend claude\|codex` в существующей теме (сбрасывает сессию) |
| Права | `AGENT_HUB_CODEX_SANDBOX` (`read-only`/`workspace-write`/`danger-full-access`, по умолчанию `workspace-write`) и `AGENT_HUB_CODEX_APPROVAL` (`untrusted`/`on-request`/`never`, по умолчанию `on-request`); одобрения всегда идут пользователю (`approvalsReviewer: user`) |

### Почему не Python SDK `openai-codex`

Проверено на 0.160.0: публичный `ApprovalMode` — только `deny_all`/`auto_review` (одобрения пользователем нет); `AsyncCodexClient` не принимает обработчик серверных запросов и одобряет всё; синхронный клиент вызывает обработчик в потоке чтения, блокируя транспорт на время ожидания кнопки; dynamic tools не регистрируются.

## Протокол (проверено на codex-cli 0.160.0)

- Рукопожатие: `initialize {clientInfo, capabilities: {experimentalApi: true}}`, затем уведомление `initialized`.
- `account/read` → `{account: null | {type: "apiKey"|"chatgpt"|...}, requiresOpenaiAuth}`.
- `account/login/start {type: "apiKey", apiKey}` → работает без сети, пишет `auth.json`.
- `thread/start {cwd, sandbox, approvalPolicy, approvalsReviewer, model?, dynamicTools}` → `{thread: {id}}`. `thread/resume {threadId, cwd, sandbox, approvalPolicy, approvalsReviewer, model?}`; dynamic tools восстанавливаются из сохранённого треда.
- `turn/start {threadId, input}` → `{turn: {id}}`; `turn/steer {threadId, expectedTurnId, input}`.
- Ввод: `{type: "text", text}`, `{type: "image", url: "data:<mime>;base64,..."}`.
- Уведомления: `item/started`, `item/completed` (`item.type`: `agentMessage`, `commandExecution`, `fileChange`, `mcpToolCall`, `webSearch`, `dynamicToolCall`, …), `thread/tokenUsage/updated` (`tokenUsage.total.totalTokens`), `turn/completed` (`turn.status`: `completed`/`interrupted`/`failed`, `turn.error.message`).
- Запросы сервера: `item/commandExecution/requestApproval` (`command`, `reason`) и `item/fileChange/requestApproval` (`reason`, `grantRoot`) → `{decision: "accept"|"decline"}`; `item/tool/call` (`tool`, `arguments`) → `{success, contentItems: [{type: "inputText", text}]}`; `item/tool/requestUserInput` (`questions[{id, header, question, options?}]`) → `{answers: {id: {answers: [..]}}}`. Прочие запросы → JSON-RPC ошибка `-32601`.

## Архитектура

### Модули

- `backends/common.py` — общее для бэкендов: `HubTool` (`send_file`, `ask_user`), описания и JSON-схемы инструментов, `ToolResult = ToolSuccess | ToolError`, `deliver_file`, `parse_questions`/`parse_option`, `TOOL_SUMMARY_LIMIT`. Claude использует их, поведение не меняется.
- `backends/rpc.py` — `RpcConnection`: JSON-RPC 2.0 построчно поверх `LineReader`/`LineWriter`. Ответы сопоставляются по id, уведомления в очередь, запросы сервера — каждый в своей задаче через обработчик, поэтому ожидание кнопки не блокирует транспорт. После закрытия транспорта все вызовы падают `TransportClosedError`.
- `backends/codex_protocol.py` — чистые функции: параметры запросов, разбор ответов, `TurnTracker.translate(notification) -> Translation(events, completed)`, `auth_state`.
- `backends/codex_requests.py` — `answer(channel, method, params)`: одобрения, `send_file`, `ask_user`, `requestUserInput`.
- `backends/codex.py` — `app_server` (процесс + соединение), `CodexBackend`, `converse`, `probe_auth`.

### Сессия

1. `app_server(handler)` запускает `codex app-server` (stdout-лимит строки 64 МиБ, stderr построчно в лог с уровнем `info`, `PATH` дополнен каталогом `codex-path` из пакета, `AGENT_HUB_TELEGRAM_TOKEN` и `OPENAI_API_KEY` из окружения убраны), отдаёт `RpcConnection`; при выходе закрывает stdin, ждёт завершения до 5 с, затем убивает процесс.
2. Рукопожатие, `account/read`: нет входа → `Failed("Codex не авторизован: …")`.
3. `thread/start` (новая сессия) или `thread/resume` (есть `session_id`) → `SessionStarted(thread_id)`. Ошибка `thread/resume` → `Failed` с подсказкой `/reset`; новый тред сам не создаётся.
4. `converse`: первый `Prompt` → `turn/start`. Prompt из inbox во время хода → `turn/steer`; если steer отклонён — сообщение ждёт завершения текущего хода и уходит новым ходом. Ход считается завершённым только по `turn/completed` с id активного хода. Сессия закрывается, когда активного хода нет и inbox пуст.
5. Ошибки: `RpcError` → `Failed("Codex: <message>")`; `TransportClosedError`/`ProtocolError`/`TimeoutError` (60 с на управляющие запросы) → `Failed("Codex app-server завершился, подробности в логе сервиса")`. Длительность хода не ограничена (только `/stop`).

### События

| Codex | Хаб |
|---|---|
| ответ `thread/start`/`thread/resume` | `SessionStarted(thread_id)` |
| `item/completed` `agentMessage` | `AssistantText` |
| `item/started` `commandExecution` | `ToolCall("shell", command)` |
| `item/started` `fileChange` | `ToolCall("patch", пути через запятую)` |
| `item/started` `mcpToolCall` | `ToolCall("<server>/<tool>", аргументы JSON)` |
| `item/started` `webSearch` | `ToolCall("web_search", query)` |
| `thread/tokenUsage/updated` | запоминается `total.totalTokens` |
| `turn/completed` `completed` | `Finished(thread_id, turns=None, cost_usd=None, tokens=…)` |
| `turn/completed` `interrupted` / иначе | `Failed` |
| прочие (`dynamicToolCall`, reasoning, дельты, …) | ничего |

### Запросы сервера

| Запрос | Обработка |
|---|---|
| одобрение команды | `ToolRequest("shell", command ∨ reason)` → `accept`/`decline` |
| одобрение правки | `ToolRequest("patch", reason ∨ «запись в <grantRoot>» ∨ «изменение файлов»)`; пути пользователь уже видел строкой `patch` |
| `item/tool/call` `send_file` | `deliver_file` |
| `item/tool/call` `ask_user` | `parse_questions` → `channel.ask` → ответы строками `вопрос: ответ` |
| `item/tool/requestUserInput` | вопросы → `channel.ask`; отказ → `{answers: {}}` |
| другое | `-32601` |

### Общий слой

- `BackendKind.CODEX = "codex"`.
- `Finished.turns: int | None`, новое `Finished.tokens: int | None = None`; `format_finished` пропускает отсутствующие части, токены — «токенов в сессии: 12 345».
- `Settings`: `default_backend` (`AGENT_HUB_DEFAULT_BACKEND`, по умолчанию `claude`), `background_timeout_seconds` (перенесён из `ClaudeSettings`), `codex: CodexSettings(model, sandbox, approval, api_key)`; `api_key` из `OPENAI_API_KEY`, скрыт из `repr`.
- `commands`: `parse_new_args(args, default)`, `parse_backend_args(args) -> BackendKind | ShowBackend | UnknownBackend`.
- `bot`: `/backend`, бэкенд по умолчанию из настроек, таймаут фона из общих настроек, токены в итоге хода, справка.
- `__main__`: `case BackendKind.CODEX`, `probe_auth` до запуска polling.

## Развёртывание

- Зависимость `openai-codex-cli-bin` (версия фиксируется в `uv.lock`); mypy читает его через `follow_untyped_imports`.
- Docker: `ENV CODEX_HOME=/home/app/.codex`, симлинк бинарника в `/usr/local/bin/codex`, том `codex`, в `compose.yaml` `AGENT_HUB_CODEX_SANDBOX: danger-full-access` (landlock/seccomp в контейнере недоступны; изоляция — сам контейнер) и `AGENT_HUB_CODEX_APPROVAL: untrusted` (без песочницы `on-request` ни о чём не спрашивает; `untrusted` спрашивает перед командами, как Claude в режиме `default`). Вход: `docker compose run --rm -it --entrypoint codex agent-hub login --device-auth`.
- CI: `docker run --rm --entrypoint codex agent-hub:ci --version`.
- README и `.env.example`: Codex, переменные, `/backend`, вход, «Что где хранится».

## Тесты

`test_common`, `test_rpc` (StreamReader + фейковый writer), `test_codex_protocol`, `test_codex_requests` (FakeChannel), `test_codex_converse` (фейковый `CodexThread`), `test_config`, `test_commands`, `test_render`; существующие тесты Claude проходят без изменений поведения. Ручная проверка с реальным входом в Codex — последней задачей.

## Вне объёма

Фоновые задачи Codex; перенос контекста между агентами при `/backend`; вход в Codex через Telegram.
