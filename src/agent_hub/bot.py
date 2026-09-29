"""Telegram shell: forum topics ↔ agent sessions."""

import asyncio
import html
import logging
from collections.abc import Coroutine, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, assert_never, final

from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter, TelegramError
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

from agent_hub import questions
from agent_hub.approvals import (
    CALLBACK_PREFIX,
    ApprovalRegistry,
    Verdict,
    callback_data,
    parse_callback_data,
)
from agent_hub.attachments import (
    MAX_DOWNLOAD_BYTES,
    MAX_SEND_BYTES,
    UPLOADS_DIR,
    AttachmentError,
    outgoing_path,
    prepare_upload,
    prompt_text,
    upload_path,
)
from agent_hub.backends import AgentBackend
from agent_hub.commands import DEFAULT_BACKEND, join_path_args, parse_new_args
from agent_hub.config import Settings
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    Answered,
    AssistantText,
    BackendKind,
    BackgroundAbandoned,
    Decision,
    Delivered,
    Denied,
    Failed,
    FileDelivery,
    Finished,
    Image,
    ImageMediaType,
    OutgoingFile,
    Prompt,
    Question,
    QuestionsOutcome,
    SessionStarted,
    ToolCall,
    ToolRequest,
    TopicKey,
    TopicSession,
)
from agent_hub.markdown_html import html_to_plain, markdown_to_html_chunks
from agent_hub.questions import (
    Accepted,
    NothingSelected,
    QuestionAnswer,
    QuestionRegistry,
    SelectionChanged,
    Stale,
    keyboard,
    question_html,
)
from agent_hub.render import format_abandoned, format_finished, split_message, truncate
from agent_hub.store import TopicStore
from agent_hub.workspace import InvalidCwdError, resolve_cwd

log = logging.getLogger(__name__)

APPROVAL_TEXT_LIMIT = 3500
TOOL_CALL_TEXT_LIMIT = 900
FAILURE_TEXT_LIMIT = 3500
ANSWER_TEXT_LIMIT = 500
CAPTION_TEXT_LIMIT = 1024  # Telegram media caption limit
# Uploads of large documents take far longer than the default request timeout.
SEND_FILE_TIMEOUT_SECONDS = 120
# Telegram delivers album parts as separate updates sharing a media_group_id.
ALBUM_WAIT_SECONDS = 1.0

HELP = """\
Каждая тема этой группы — отдельная сессия агента.

Просто пишите задачу в теме. Команды:
/new [backend] [путь] — новая сессия в этой теме (сброс контекста)
/cwd <путь> — сменить рабочую директорию (сброс контекста)
/reset — начать разговор заново в той же директории
/stop — прервать текущую задачу
/status — состояние сессии
/help — эта справка

Можно прикладывать фото (агент их видит) и файлы (сохраняются в {uploads} в
рабочей директории). Голосовые сообщения не поддерживаются.

Пути абсолютные или относительно корня: {root}
Бэкенды: {backends}"""


@final
@dataclass(frozen=True, slots=True)
class Upload:
    message_id: int
    file_id: str
    filename: str | None
    size: int | None


@final
@dataclass(frozen=True, slots=True)
class Incoming:
    """A user turn as received from Telegram, before attachments are downloaded."""

    text: str
    photos: tuple[Upload, ...] = ()
    files: tuple[Upload, ...] = ()


@final
@dataclass(frozen=True, slots=True)
class LiveSession:
    """An open agent session of a topic; messages sent meanwhile go to its inbox."""

    task: "asyncio.Task[None]"
    inbox: "asyncio.Queue[Prompt]"


class TelegramSender:
    """Best-effort delivery: a lost chat message must not abort the agent run."""

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def text(self, key: TopicKey, text: str) -> None:
        for chunk in split_message(text):
            await self.one(key, chunk)

    async def markdown(self, key: TopicKey, markdown: str) -> None:
        for chunk in markdown_to_html_chunks(markdown):
            await self.one(key, chunk, parse_mode=ParseMode.HTML)

    async def one(
        self,
        key: TopicKey,
        text: str,
        markup: InlineKeyboardMarkup | None = None,
        parse_mode: ParseMode | None = None,
    ) -> Message | None:
        for attempt in range(2):
            try:
                return await self._bot.send_message(
                    chat_id=key.chat_id,
                    message_thread_id=key.thread_id,
                    text=text,
                    reply_markup=markup,
                    parse_mode=parse_mode,
                )
            except RetryAfter as error:
                if attempt == 1:
                    log.warning("telegram rate limit persisted", extra=_fields(key))
                    return None
                await asyncio.sleep(_seconds(error.retry_after))
            except BadRequest as error:
                if parse_mode is None:
                    log.exception("telegram send failed", extra=_fields(key))
                    return None
                # A formatting bug must not lose the message: resend it as plain text.
                log.warning(
                    "telegram rejected formatting, sending plain text",
                    extra={**_fields(key), "error": str(error)},
                )
                return await self.one(key, html_to_plain(text), markup)
            except TelegramError:
                log.exception("telegram send failed", extra=_fields(key))
                return None
        return None

    async def edit(self, message: Message, text: str, parse_mode: ParseMode | None = None) -> None:
        try:
            await message.edit_text(text, reply_markup=None, parse_mode=parse_mode)
        except TelegramError:
            log.exception("telegram edit failed", extra={"message_id": message.message_id})

    async def document(self, key: TopicKey, path: Path, caption: str) -> None:
        """Unlike text, a failed file delivery is raised: the agent must learn about it."""
        for attempt in range(2):
            try:
                await self._bot.send_document(
                    chat_id=key.chat_id,
                    message_thread_id=key.thread_id,
                    document=path,
                    filename=path.name,
                    caption=truncate(caption, CAPTION_TEXT_LIMIT) or None,
                    write_timeout=SEND_FILE_TIMEOUT_SECONDS,
                    read_timeout=SEND_FILE_TIMEOUT_SECONDS,
                )
            except RetryAfter as error:
                if attempt == 1:
                    raise
                await asyncio.sleep(_seconds(error.retry_after))
            else:
                return

    async def typing(self, key: TopicKey) -> None:
        try:
            await self._bot.send_chat_action(
                chat_id=key.chat_id, message_thread_id=key.thread_id, action=ChatAction.TYPING
            )
        except TelegramError:
            log.warning("telegram chat action failed", extra=_fields(key))


@final
@dataclass(frozen=True, slots=True)
class Pending:
    """Hub-wide registries of prompts waiting for a button press or reply."""

    approvals: ApprovalRegistry
    questions: QuestionRegistry


class TelegramChannel:
    def __init__(
        self, sender: TelegramSender, key: TopicKey, cwd: Path, pending: Pending, timeout: int
    ) -> None:
        self._sender = sender
        self._key = key
        self._cwd = cwd
        self._registry = pending.approvals
        self._questions = pending.questions
        self._timeout = timeout

    async def send_file(self, file: OutgoingFile) -> FileDelivery:
        try:
            path = await asyncio.to_thread(outgoing_path, self._cwd, file.path, MAX_SEND_BYTES)
            await self._sender.document(self._key, path, file.caption)
        except AttachmentError as error:
            return Denied(str(error))
        except (TelegramError, OSError) as error:
            log.warning("file delivery failed", extra={**_fields(self._key), "error": str(error)})
            return Denied(f"Telegram не принял файл: {error}")
        log.info("file sent", extra={**_fields(self._key), "file": path.name})
        return Delivered()

    async def ask(self, questions: Sequence[Question]) -> QuestionsOutcome:
        answers: list[tuple[str, str]] = []
        for question in questions:
            answer = await self._ask_one(question)
            if isinstance(answer, Denied):
                return answer
            answers.append((question.text, answer))
        return Answered(tuple(answers))

    async def _ask_one(self, question: Question) -> QuestionAnswer:
        question_id, future = self._questions.open(self._key, question)
        try:
            text = question_html(question)
            markup = _markup(keyboard(question_id, question, frozenset()))
            prompt = await self._sender.one(self._key, text, markup, ParseMode.HTML)
            if prompt is None:
                return Denied("Не удалось отправить вопрос в Telegram")
            try:
                answer = await asyncio.wait_for(future, self._timeout)
            except TimeoutError:
                await self._sender.edit(prompt, f"{text}\n\n⌛ Нет ответа", ParseMode.HTML)
                return Denied(f"Нет ответа пользователя за {self._timeout} с")
            await self._sender.edit(prompt, f"{text}\n\n{_answer_line(answer)}", ParseMode.HTML)
            return answer
        finally:
            self._questions.close(question_id)

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
            summary = html.escape(truncate(tool.summary, APPROVAL_TEXT_LIMIT))
            text = f"🔐 <b>{html.escape(tool.tool)}</b>\n<pre>{summary}</pre>"
            prompt = await self._sender.one(self._key, text, markup, ParseMode.HTML)
            if prompt is None:
                return Denied("Не удалось отправить запрос подтверждения в Telegram")
            try:
                return await asyncio.wait_for(future, self._timeout)
            except TimeoutError:
                await self._sender.edit(
                    prompt, f"{text}\n\n⌛ Нет ответа — запрещено", ParseMode.HTML
                )
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
        self._questions = QuestionRegistry()
        self._pending = Pending(self._approvals, self._questions)
        self._running: dict[TopicKey, LiveSession] = {}
        self._albums: dict[str, list[Message]] = {}
        # Unawaited helper tasks (album flushes, attachment downloads), kept to cancel on stop.
        self._helpers: set[asyncio.Task[None]] = set()

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
            CallbackQueryHandler(self._question, pattern=f"^{questions.CALLBACK_PREFIX}:")
        )
        app.add_handler(
            MessageHandler(fresh & filters.StatusUpdate.FORUM_TOPIC_CREATED, self._topic_created)
        )
        app.add_handler(MessageHandler(fresh & filters.TEXT & ~filters.COMMAND, self._text))
        media = filters.PHOTO | filters.Document.ALL | filters.AUDIO | filters.VIDEO
        app.add_handler(MessageHandler(fresh & media, self._media))
        app.add_handler(
            MessageHandler(fresh & (filters.VOICE | filters.VIDEO_NOTE), self._unsupported)
        )
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
        if key is None or await self._refuse_if_running(key, context.bot):
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
        if key is None or await self._refuse_if_running(key, context.bot):
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
        if key is None or await self._refuse_if_running(key, context.bot):
            return
        session = self._session(key).with_session(None)
        self._store.put(key, session)
        await TelegramSender(context.bot).text(key, _describe("🔄 Контекст сброшен", session))

    async def _stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None:
            return
        live = self._running.get(key)
        if live is None:
            await TelegramSender(context.bot).text(key, "Нечего останавливать")
            return
        live.task.cancel()

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
                query.message, f"{query.message.text_html}\n\n{verdict}", ParseMode.HTML
            )

    async def _question(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        press = questions.parse_callback_data(query.data or "")
        if press is None:
            await query.answer("Вопрос уже неактуален")
            return
        result = self._questions.press(press)
        match result:
            case Accepted():
                # The asking channel edits the message once it sees the answer.
                await query.answer()
            case SelectionChanged(question, selected):
                await query.answer()
                try:
                    await query.edit_message_reply_markup(
                        _markup(keyboard(press.question_id, question, selected))
                    )
                except TelegramError:
                    log.warning("telegram markup edit failed", exc_info=True)
            case NothingSelected():
                await query.answer("Отметьте хотя бы один вариант")
            case Stale():
                await query.answer("Вопрос уже неактуален")
            case _:
                assert_never(result)

    async def _text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        key = await self._topic_or_hint(update)
        if key is None or message is None or not message.text:
            return
        # While the agent waits on a question, the next message in the topic is its answer.
        if self._questions.reply(key, message.text):
            return
        await self._start_turn(context.bot, key, Incoming(message.text))

    async def _media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        key = await self._topic_or_hint(update)
        if key is None or message is None:
            return
        group = message.media_group_id
        if group is None:
            await self._start_turn(context.bot, key, _incoming([message]))
            return
        parts = self._albums.setdefault(group, [])
        parts.append(message)
        if len(parts) == 1:
            self._helper(self._flush_album(context.bot, key, group))

    async def _flush_album(self, bot: Bot, key: TopicKey, group: str) -> None:
        await asyncio.sleep(ALBUM_WAIT_SECONDS)
        parts = self._albums.pop(group, [])
        if not parts:
            return
        try:
            await self._start_turn(bot, key, _incoming(parts))
        except Exception:  # background task boundary: nobody awaits this task
            log.exception("album dispatch failed", extra={**_fields(key), "media_group": group})

    async def _unsupported(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is not None:
            await TelegramSender(context.bot).text(
                key, "🎤 Голосовые сообщения пока не поддерживаются — напишите текстом."
            )

    async def _start_turn(self, bot: Bot, key: TopicKey, incoming: Incoming) -> None:
        live = self._running.get(key)
        if live is None:
            self._launch(bot, key, incoming)
        elif incoming.photos or incoming.files:
            self._helper(self._forward(bot, key, incoming))
        else:
            live.inbox.put_nowait(Prompt(incoming.text))

    async def _forward(self, bot: Bot, key: TopicKey, incoming: Incoming) -> None:
        """Attachments sent to an open session: download first, then deliver."""
        sender = TelegramSender(bot)
        try:
            prompt = await _download(bot, self._session(key).cwd, incoming)
        except AttachmentError as error:
            log.warning("attachment rejected", extra={**_fields(key), "reason": str(error)})
            await sender.text(key, f"⚠️ {error}")
            return
        # The session may have closed during the download.
        live = self._running.get(key)
        if live is None:
            self._launch(bot, key, prompt)
        else:
            live.inbox.put_nowait(prompt)

    def _launch(self, bot: Bot, key: TopicKey, first: Incoming | Prompt) -> None:
        inbox: asyncio.Queue[Prompt] = asyncio.Queue()
        task = asyncio.create_task(
            self._run_session(bot, key, first, inbox),
            name=f"session-{key.chat_id}-{key.thread_id}",
        )
        self._running[key] = LiveSession(task, inbox)

    def _helper(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(self._guarded(work))
        self._helpers.add(task)
        task.add_done_callback(self._helpers.discard)

    async def _guarded(self, work: Coroutine[Any, Any, None]) -> None:
        try:
            await work
        except Exception:  # background task boundary: nobody awaits this task
            log.exception("helper task failed")

    # --- agent run ------------------------------------------------------------

    async def _run_session(
        self, bot: Bot, key: TopicKey, first: Incoming | Prompt, inbox: "asyncio.Queue[Prompt]"
    ) -> None:
        sender = TelegramSender(bot)
        live = self._running[key]
        log.info("session started", extra=_fields(key))
        try:
            await sender.typing(key)
            if isinstance(first, Incoming):
                try:
                    first = await _download(bot, self._session(key).cwd, first)
                except AttachmentError as error:
                    log.warning("attachment rejected", extra={**_fields(key), "reason": str(error)})
                    await sender.text(key, f"⚠️ {error}")
                    return
            prompt = first
            while True:
                await self._converse(sender, key, prompt, inbox)
                # No await between this check and the release: a message arriving now must
                # either be seen here or start a new session.
                if inbox.empty():
                    self._release(key, live)
                    return
                prompt = inbox.get_nowait()
        except asyncio.CancelledError:
            self._release(key, live)
            log.info("session cancelled", extra=_fields(key))
            await sender.text(key, "⏹ Остановлено")
            raise
        except Exception:  # background task boundary: record and report
            self._release(key, live)
            log.exception("session crashed", extra=_fields(key))
            await sender.text(key, "💥 Внутренняя ошибка agent-hub, подробности в логе сервиса")
        finally:
            self._release(key, live)

    async def _converse(
        self, sender: TelegramSender, key: TopicKey, prompt: Prompt, inbox: "asyncio.Queue[Prompt]"
    ) -> None:
        session = self._session(key)
        channel = TelegramChannel(
            sender, key, session.cwd, self._pending, self._settings.approval_timeout_seconds
        )
        backend = self._backends[session.backend]
        async with aclosing(backend.run(session, prompt, channel, inbox)) as events:
            async for event in events:
                await self._on_event(sender, key, event)

    def _release(self, key: TopicKey, live: LiveSession) -> None:
        if self._running.get(key) is live:
            del self._running[key]

    async def _on_event(self, sender: TelegramSender, key: TopicKey, event: AgentEvent) -> None:
        match event:
            case SessionStarted(session_id):
                # Persist immediately so a crash mid-turn can still resume this session.
                self._store.put(key, self._session(key).with_session(session_id))
            case AssistantText(text):
                await sender.markdown(key, text)
            case ToolCall(tool, summary):
                line = html.escape(truncate(summary, TOOL_CALL_TEXT_LIMIT))
                await sender.one(
                    key,
                    f"🔧 <b>{html.escape(tool)}</b> <code>{line}</code>",
                    parse_mode=ParseMode.HTML,
                )
            case Finished(session_id, turns, cost_usd, background):
                self._store.put(key, self._session(key).with_session(session_id))
                log.info(
                    "turn finished",
                    extra={**_fields(key), "turns": turns, "background": background},
                )
                await sender.text(key, format_finished(turns, cost_usd, background))
            case BackgroundAbandoned(tasks):
                log.warning("background abandoned", extra={**_fields(key), "tasks": len(tasks)})
                timeout = self._settings.claude.background_timeout_seconds
                await sender.text(key, format_abandoned(tasks, timeout))
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

    async def _refuse_if_running(self, key: TopicKey, bot: Bot) -> bool:
        if key not in self._running:
            return False
        await TelegramSender(bot).text(
            key, "⏳ В этой теме уже выполняется задача. /stop — прервать."
        )
        return True

    async def _cancel_all(self, _: Application[Any, Any, Any, Any, Any, Any]) -> None:
        tasks = [*(live.task for live in self._running.values()), *self._helpers]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _help_text(self) -> str:
        return HELP.format(
            uploads=UPLOADS_DIR,
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


def _incoming(messages: Sequence[Message]) -> Incoming:
    """One turn from a message or an album: captions joined, attachments in message order."""
    text = "\n".join(m.text or m.caption or "" for m in messages if m.text or m.caption)
    photos = tuple(
        Upload(m.message_id, m.photo[-1].file_id, None, m.photo[-1].file_size)
        for m in messages
        if m.photo
    )
    files = tuple(
        Upload(m.message_id, media.file_id, media.file_name, media.file_size)
        for m in messages
        for media in (m.document, m.audio, m.video)
        if media is not None
    )
    return Incoming(text, photos, files)


async def _download(bot: Bot, cwd: Path, incoming: Incoming) -> Prompt:
    """Fetch attachments: photos inline for the model, other files into the session cwd."""
    for upload in (*incoming.photos, *incoming.files):
        if upload.size is not None and upload.size > MAX_DOWNLOAD_BYTES:
            raise AttachmentError(
                f"{upload.filename or 'фото'}: больше {MAX_DOWNLOAD_BYTES // 2**20} МБ, "
                "Telegram не отдаёт боту такие файлы"
            )
    try:
        # Telegram re-encodes every photo it stores as JPEG.
        images = [
            Image(
                ImageMediaType.JPEG,
                bytes(await (await bot.get_file(photo.file_id)).download_as_bytearray()),
            )
            for photo in incoming.photos
        ]
        paths = [await _store(bot, cwd, upload) for upload in incoming.files]
    except TelegramError as error:
        raise AttachmentError(f"Не удалось скачать вложение: {error}") from error
    except OSError as error:
        raise AttachmentError(f"Не удалось сохранить вложение: {error}") from error
    return Prompt(prompt_text(incoming.text, paths), tuple(images))


async def _fetch_photo(bot: Bot, photo: Upload) -> Image:
    file = await bot.get_file(photo.file_id)
    # Telegram re-encodes every photo it stores as JPEG.
    return Image(ImageMediaType.JPEG, bytes(await file.download_as_bytearray()))


async def _store(bot: Bot, cwd: Path, upload: Upload) -> Path:
    path = upload_path(cwd, upload.message_id, upload.filename)
    await asyncio.to_thread(prepare_upload, cwd, path)
    file = await bot.get_file(upload.file_id)
    return await file.download_to_drive(path)


def _markup(rows: Sequence[Sequence[questions.Button]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=data) for text, data in row] for row in rows]
    )


def _answer_line(answer: QuestionAnswer) -> str:
    match answer:
        case str():
            return f"💬 {html.escape(truncate(answer, ANSWER_TEXT_LIMIT))}"
        case Denied():
            return "❌ Без ответа"
        case _:
            assert_never(answer)
