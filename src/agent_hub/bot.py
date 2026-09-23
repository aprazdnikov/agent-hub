"""Telegram shell: forum topics ↔ agent sessions."""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import aclosing
from datetime import timedelta
from typing import Any, assert_never

from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import RetryAfter, TelegramError
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from agent_hub.approvals import (
    CALLBACK_PREFIX,
    ApprovalRegistry,
    Verdict,
    callback_data,
    parse_callback_data,
)
from agent_hub.backends import AgentBackend
from agent_hub.commands import DEFAULT_BACKEND, join_path_args, parse_new_args
from agent_hub.config import Settings
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    AssistantText,
    BackendKind,
    Decision,
    Denied,
    Failed,
    Finished,
    SessionStarted,
    ToolCall,
    ToolRequest,
    TopicKey,
    TopicSession,
)
from agent_hub.render import format_finished, split_message, truncate
from agent_hub.store import TopicStore
from agent_hub.workspace import InvalidCwdError, resolve_cwd

log = logging.getLogger(__name__)

APPROVAL_TEXT_LIMIT = 3500
FAILURE_TEXT_LIMIT = 3500

HELP = """\
Каждая тема этой группы — отдельная сессия агента.

Просто пишите задачу в теме. Команды:
/new [backend] [путь] — новая сессия в этой теме (сброс контекста)
/cwd <путь> — сменить рабочую директорию (сброс контекста)
/reset — начать разговор заново в той же директории
/stop — прервать текущую задачу
/status — состояние сессии
/help — эта справка

Пути абсолютные или относительно корня: {root}
Бэкенды: {backends}"""


class TelegramSender:
    """Best-effort delivery: a lost chat message must not abort the agent run."""

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def text(self, key: TopicKey, text: str) -> None:
        for chunk in split_message(text):
            await self.one(key, chunk)

    async def one(
        self, key: TopicKey, text: str, markup: InlineKeyboardMarkup | None = None
    ) -> Message | None:
        for attempt in range(2):
            try:
                return await self._bot.send_message(
                    chat_id=key.chat_id,
                    message_thread_id=key.thread_id,
                    text=text,
                    reply_markup=markup,
                )
            except RetryAfter as error:
                if attempt == 1:
                    log.warning("telegram rate limit persisted", extra=_fields(key))
                    return None
                await asyncio.sleep(_seconds(error.retry_after))
            except TelegramError:
                log.exception("telegram send failed", extra=_fields(key))
                return None
        return None

    async def edit(self, message: Message, text: str) -> None:
        try:
            await message.edit_text(text, reply_markup=None)
        except TelegramError:
            log.exception("telegram edit failed", extra={"message_id": message.message_id})

    async def typing(self, key: TopicKey) -> None:
        try:
            await self._bot.send_chat_action(
                chat_id=key.chat_id, message_thread_id=key.thread_id, action=ChatAction.TYPING
            )
        except TelegramError:
            log.warning("telegram chat action failed", extra=_fields(key))


class TelegramApprovalGate:
    def __init__(
        self, sender: TelegramSender, key: TopicKey, registry: ApprovalRegistry, timeout: int
    ) -> None:
        self._sender = sender
        self._key = key
        self._registry = registry
        self._timeout = timeout

    async def request(self, tool: ToolRequest) -> Decision:
        approval_id, future = self._registry.open()
        try:
            markup = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Разрешить", callback_data=callback_data(approval_id, Verdict.ALLOW)
                        ),
                        InlineKeyboardButton(
                            "❌ Запретить", callback_data=callback_data(approval_id, Verdict.DENY)
                        ),
                    ]
                ]
            )
            text = truncate(f"🔐 {tool.tool}\n{tool.summary}", APPROVAL_TEXT_LIMIT)
            prompt = await self._sender.one(self._key, text, markup)
            if prompt is None:
                return Denied("Не удалось отправить запрос подтверждения в Telegram")
            try:
                return await asyncio.wait_for(future, self._timeout)
            except TimeoutError:
                await self._sender.edit(prompt, f"{text}\n\n⌛ Нет ответа — запрещено")
                return Denied(f"Нет ответа пользователя за {self._timeout} с")
        finally:
            self._registry.close(approval_id)


class Hub:
    def __init__(
        self,
        settings: Settings,
        store: TopicStore,
        backends: Mapping[BackendKind, AgentBackend],
    ) -> None:
        self._settings = settings
        self._store = store
        self._backends = backends
        self._approvals = ApprovalRegistry()
        self._running: dict[TopicKey, asyncio.Task[None]] = {}

    def build_application(self) -> Application[Any, Any, Any, Any, Any, Any]:
        app = (
            Application.builder()
            .token(self._settings.telegram_token)
            .post_stop(self._cancel_all)
            .build()
        )
        # Edits of old messages must not re-run commands or start new turns.
        fresh = filters.UpdateType.MESSAGE
        app.add_handler(TypeHandler(Update, self._guard), group=-1)
        app.add_handler(CommandHandler(["start", "help"], self._help, filters=fresh))
        app.add_handler(CommandHandler("new", self._new, filters=fresh))
        app.add_handler(CommandHandler("cwd", self._cwd, filters=fresh))
        app.add_handler(CommandHandler("reset", self._reset, filters=fresh))
        app.add_handler(CommandHandler("stop", self._stop, filters=fresh))
        app.add_handler(CommandHandler("status", self._status, filters=fresh))
        app.add_handler(CallbackQueryHandler(self._approval, pattern=f"^{CALLBACK_PREFIX}:"))
        app.add_handler(
            MessageHandler(fresh & filters.StatusUpdate.FORUM_TOPIC_CREATED, self._topic_created)
        )
        app.add_handler(MessageHandler(fresh & filters.TEXT & ~filters.COMMAND, self._text))
        return app

    # --- handlers -------------------------------------------------------------

    async def _guard(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        user, chat = update.effective_user, update.effective_chat
        if (
            user is not None
            and chat is not None
            and user.id in self._settings.allowed_user_ids
            and chat.id == self._settings.chat_id
        ):
            return
        log.warning(
            "rejected update",
            extra={
                "user_id": None if user is None else user.id,
                "chat_id": None if chat is None else chat.id,
            },
        )
        raise ApplicationHandlerStop

    async def _help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is not None:
            await message.reply_text(self._help_text())

    async def _topic_created(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = _topic_key(update.effective_message)
        if key is None:
            return
        session = TopicSession(DEFAULT_BACKEND, self._settings.workspace_root, None)
        self._store.put(key, session)
        log.info("topic bound", extra=_fields(key))
        await TelegramSender(context.bot).text(key, _describe("🆕 Новая сессия", session))

    async def _new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None or await self._refuse_if_running(key, context):
            return
        args = parse_new_args(context.args or [])
        try:
            cwd = resolve_cwd(self._settings.workspace_root, args.cwd)
        except InvalidCwdError as error:
            await TelegramSender(context.bot).text(key, f"⚠️ {error}")
            return
        session = TopicSession(args.backend, cwd, None)
        self._store.put(key, session)
        await TelegramSender(context.bot).text(key, _describe("🆕 Новая сессия", session))

    async def _cwd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None or await self._refuse_if_running(key, context):
            return
        raw = join_path_args(context.args or [])
        sender = TelegramSender(context.bot)
        if raw is None:
            await sender.text(key, "Укажите путь: /cwd <путь>")
            return
        try:
            cwd = resolve_cwd(self._settings.workspace_root, raw)
        except InvalidCwdError as error:
            await sender.text(key, f"⚠️ {error}")
            return
        current = self._session(key)
        # Claude sessions are stored per project directory, so a new cwd needs a new session.
        session = TopicSession(current.backend, cwd, None)
        self._store.put(key, session)
        await sender.text(key, _describe("📁 Директория изменена", session))

    async def _reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None or await self._refuse_if_running(key, context):
            return
        session = self._session(key).with_session(None)
        self._store.put(key, session)
        await TelegramSender(context.bot).text(key, _describe("🔄 Контекст сброшен", session))

    async def _stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None:
            return
        task = self._running.get(key)
        if task is None:
            await TelegramSender(context.bot).text(key, "Нечего останавливать")
            return
        task.cancel()

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None:
            return
        session = self._store.get(key)
        if session is None:
            await TelegramSender(context.bot).text(key, "Сессии нет — напишите задачу или /new")
            return
        state = "⏳ выполняется" if key in self._running else "💤 ожидает"
        await TelegramSender(context.bot).text(key, _describe(state, session))

    async def _approval(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        answer = parse_callback_data(query.data or "")
        if answer is None or not self._approvals.resolve(answer):
            await query.answer("Запрос уже неактуален")
            return
        await query.answer()
        match answer.decision:
            case Allowed():
                verdict = "✅ Разрешено"
            case Denied():
                verdict = "❌ Запрещено"
            case _:
                assert_never(answer.decision)
        if isinstance(query.message, Message):
            await TelegramSender(query.get_bot()).edit(
                query.message, f"{query.message.text or ''}\n\n{verdict}"
            )

    async def _text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        key = await self._topic_or_hint(update)
        if key is None or message is None or not message.text:
            return
        if await self._refuse_if_running(key, context):
            return
        session = self._session(key)
        task = asyncio.create_task(
            self._run_turn(TelegramSender(context.bot), key, session, message.text),
            name=f"turn-{key.chat_id}-{key.thread_id}",
        )
        self._running[key] = task
        task.add_done_callback(lambda done: self._forget(key, done))

    # --- agent run ------------------------------------------------------------

    async def _run_turn(
        self, sender: TelegramSender, key: TopicKey, session: TopicSession, prompt: str
    ) -> None:
        backend = self._backends[session.backend]
        gate = TelegramApprovalGate(
            sender, key, self._approvals, self._settings.approval_timeout_seconds
        )
        log.info("turn started", extra={**_fields(key), "backend": session.backend.value})
        try:
            await sender.typing(key)
            async with aclosing(backend.run(session, prompt, gate)) as events:
                async for event in events:
                    await self._on_event(sender, key, event)
        except asyncio.CancelledError:
            log.info("turn cancelled", extra=_fields(key))
            await sender.text(key, "⏹ Остановлено")
            raise
        except Exception:  # background task boundary: record and report
            log.exception("turn crashed", extra=_fields(key))
            await sender.text(key, "💥 Внутренняя ошибка agent-hub, подробности в логе сервиса")

    async def _on_event(self, sender: TelegramSender, key: TopicKey, event: AgentEvent) -> None:
        match event:
            case SessionStarted(session_id):
                # Persist immediately so a crash mid-turn can still resume this session.
                self._store.put(key, self._session(key).with_session(session_id))
            case AssistantText(text):
                await sender.text(key, text)
            case ToolCall(tool, summary):
                await sender.text(key, truncate(f"🔧 {tool}: {summary}", 1000))
            case Finished(session_id, turns, cost_usd):
                self._store.put(key, self._session(key).with_session(session_id))
                log.info("turn finished", extra={**_fields(key), "turns": turns})
                await sender.text(key, format_finished(turns, cost_usd))
            case Failed(reason):
                log.warning("turn failed", extra={**_fields(key), "reason": reason})
                await sender.text(key, truncate(f"❌ {reason}", FAILURE_TEXT_LIMIT))
            case _:
                assert_never(event)

    # --- helpers --------------------------------------------------------------

    def _session(self, key: TopicKey) -> TopicSession:
        """Session bound to the topic, binding the default one on first use."""
        session = self._store.get(key)
        if session is None:
            session = TopicSession(DEFAULT_BACKEND, self._settings.workspace_root, None)
            self._store.put(key, session)
            log.info("topic bound", extra=_fields(key))
        return session

    async def _topic_or_hint(self, update: Update) -> TopicKey | None:
        message = update.effective_message
        key = _topic_key(message)
        if key is None and message is not None:
            await message.reply_text("Создайте тему в группе — каждая тема это отдельная сессия.")
        return key

    async def _refuse_if_running(self, key: TopicKey, context: ContextTypes.DEFAULT_TYPE) -> bool:
        if key not in self._running:
            return False
        await TelegramSender(context.bot).text(
            key, "⏳ В этой теме уже выполняется задача. /stop — прервать."
        )
        return True

    def _forget(self, key: TopicKey, task: asyncio.Task[None]) -> None:
        if self._running.get(key) is task:
            del self._running[key]

    async def _cancel_all(self, _: Application[Any, Any, Any, Any, Any, Any]) -> None:
        tasks = list(self._running.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _help_text(self) -> str:
        return HELP.format(
            root=self._settings.workspace_root,
            backends=", ".join(kind.value for kind in BackendKind),
        )


def _topic_key(message: Message | None) -> TopicKey | None:
    if message is None or not message.is_topic_message or message.message_thread_id is None:
        return None
    return TopicKey(message.chat_id, message.message_thread_id)


def _describe(title: str, session: TopicSession) -> str:
    return (
        f"{title}\n"
        f"backend: {session.backend.value}\n"
        f"cwd: {session.cwd}\n"
        f"session: {session.session_id or '—'}"
    )


def _fields(key: TopicKey) -> dict[str, int]:
    return {"chat_id": key.chat_id, "thread_id": key.thread_id}


def _seconds(retry_after: int | timedelta) -> float:
    return retry_after.total_seconds() if isinstance(retry_after, timedelta) else retry_after
