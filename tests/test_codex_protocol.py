import base64
from pathlib import Path
from typing import Any

import pytest

from agent_hub.backends.codex_protocol import (
    Call,
    CodexAuth,
    Translation,
    TurnId,
    TurnTracker,
    auth_state,
    hub_tools,
    open_thread,
    parse_thread_id,
    parse_turn_id,
    steer_params,
    user_input,
)
from agent_hub.backends.rpc import Notification, ProtocolError
from agent_hub.config import CodexApproval, CodexSandbox, CodexSettings
from agent_hub.domain import (
    AssistantText,
    BackendKind,
    Failed,
    Finished,
    Image,
    ImageMediaType,
    Prompt,
    SessionId,
    ToolCall,
    TopicSession,
)

THREAD = SessionId("th-1")
SETTINGS = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, None)


def _session(cwd: Path, session_id: SessionId | None) -> TopicSession:
    return TopicSession(BackendKind.CODEX, cwd, session_id)


def _item(event: str, item: dict[str, Any]) -> Notification:
    return Notification(event, {"threadId": THREAD, "turnId": "turn-1", "item": item})


def _completed(turn: dict[str, Any]) -> Notification:
    return Notification("turn/completed", {"threadId": THREAD, "turn": turn})


def test_new_session_starts_a_thread_with_hub_tools(tmp_path: Path) -> None:
    method, params = open_thread(_session(tmp_path, None), SETTINGS)

    assert method is Call.THREAD_START
    assert params == {
        "cwd": str(tmp_path),
        "sandbox": "workspace-write",
        "approvalPolicy": "on-request",
        "approvalsReviewer": "user",
        "dynamicTools": hub_tools(),
    }
    assert [tool["name"] for tool in hub_tools()] == ["send_file", "ask_user"]


def test_saved_session_resumes_its_thread_with_the_model(tmp_path: Path) -> None:
    settings = CodexSettings("gpt-6.1-sol", CodexSandbox.READ_ONLY, CodexApproval.NEVER, None)

    method, params = open_thread(_session(tmp_path, THREAD), settings)

    assert method is Call.THREAD_RESUME
    assert params == {
        "cwd": str(tmp_path),
        "sandbox": "read-only",
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "model": "gpt-6.1-sol",
        "threadId": "th-1",
    }


def test_ids_are_read_from_responses() -> None:
    assert parse_thread_id({"thread": {"id": "th-1", "preview": ""}}) == THREAD
    assert parse_turn_id({"turn": {"id": "turn-1", "items": []}}) == TurnId("turn-1")
    with pytest.raises(ProtocolError):
        parse_thread_id({"thread": {}})
    with pytest.raises(ProtocolError):
        parse_turn_id({})


def test_images_go_before_text_as_data_urls() -> None:
    prompt = Prompt("что на фото?", (Image(ImageMediaType.PNG, b"\x89P"),))

    assert user_input(prompt) == [
        {"type": "image", "url": f"data:image/png;base64,{base64.b64encode(b'\x89P').decode()}"},
        {"type": "text", "text": "что на фото?"},
    ]
    assert [item["type"] for item in user_input(Prompt("", prompt.images))] == ["image"]


def test_steer_names_the_expected_turn() -> None:
    assert steer_params(THREAD, TurnId("turn-1"), Prompt("ещё")) == {
        "threadId": "th-1",
        "expectedTurnId": "turn-1",
        "input": [{"type": "text", "text": "ещё"}],
    }


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            {"type": "commandExecution", "id": "i", "command": "npm test", "cwd": "/w"},
            ToolCall("shell", "npm test"),
        ),
        (
            {
                "type": "fileChange",
                "id": "i",
                "status": "inProgress",
                "changes": [
                    {"path": "a.py", "kind": "add", "diff": ""},
                    {"path": "b.py", "kind": "update", "diff": ""},
                ],
            },
            ToolCall("patch", "a.py, b.py"),
        ),
        (
            {"type": "fileChange", "id": "i", "status": "inProgress", "changes": [{"kind": "add"}]},
            ToolCall("patch", "изменение файлов"),
        ),
        (
            {
                "type": "mcpToolCall",
                "id": "i",
                "server": "github",
                "tool": "get_issue",
                "arguments": {"n": 1},
                "status": "inProgress",
            },
            ToolCall("github/get_issue", '{"n": 1}'),
        ),
        ({"type": "webSearch", "id": "i", "query": "codex"}, ToolCall("web_search", "codex")),
    ],
)
def test_started_tool_items_become_tool_calls(item: dict[str, Any], expected: ToolCall) -> None:
    assert TurnTracker(THREAD).translate(_item("item/started", item)) == Translation((expected,))


@pytest.mark.parametrize(
    "item",
    [
        {"type": "dynamicToolCall", "id": "i", "tool": "send_file", "arguments": {}},
        {"type": "reasoning", "id": "i"},
        {"type": "agentMessage", "id": "i", "text": "ещё пишу"},
    ],
)
def test_other_started_items_are_silent(item: dict[str, Any]) -> None:
    assert TurnTracker(THREAD).translate(_item("item/started", item)) == Translation()


def test_completed_agent_message_becomes_text() -> None:
    tracker = TurnTracker(THREAD)
    message = {"type": "agentMessage", "id": "i", "text": "Готово"}

    assert tracker.translate(_item("item/completed", message)) == Translation(
        (AssistantText("Готово"),)
    )
    blank = {"type": "agentMessage", "id": "i", "text": "  "}
    assert tracker.translate(_item("item/completed", blank)) == Translation()


def test_completed_turn_finishes_with_thread_tokens() -> None:
    tracker = TurnTracker(THREAD)
    usage = {"total": {"totalTokens": 12345}, "last": {"totalTokens": 10}}
    tracker.translate(Notification("thread/tokenUsage/updated", {"tokenUsage": usage}))

    translation = tracker.translate(_completed({"id": "turn-1", "status": "completed"}))

    assert translation == Translation(
        (Finished(THREAD, None, None, tokens=12345),), TurnId("turn-1")
    )


@pytest.mark.parametrize(
    ("turn", "reason"),
    [
        ({"id": "turn-1", "status": "interrupted"}, "Ход Codex прерван"),
        ({"id": "turn-1", "status": "failed", "error": {"message": "401"}}, "401"),
        ({"id": "turn-1", "status": "failed"}, "Ход Codex завершился ошибкой"),
    ],
)
def test_unsuccessful_turn_fails(turn: dict[str, Any], reason: str) -> None:
    assert TurnTracker(THREAD).translate(_completed(turn)) == Translation(
        (Failed(reason),), TurnId("turn-1")
    )


def test_completion_without_a_turn_is_a_protocol_error() -> None:
    with pytest.raises(ProtocolError):
        TurnTracker(THREAD).translate(Notification("turn/completed", {}))


def test_unknown_notifications_are_ignored() -> None:
    notice = Notification("account/rateLimits/updated", {"x": 1})
    assert TurnTracker(THREAD).translate(notice) == Translation()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"account": {"type": "apiKey"}, "requiresOpenaiAuth": True}, CodexAuth.API_KEY),
        (
            {"account": {"type": "chatgpt", "email": None, "planType": "plus"}},
            CodexAuth.CHATGPT,
        ),
        ({"account": {"type": "amazonBedrock"}, "requiresOpenaiAuth": False}, CodexAuth.OTHER),
        ({"account": None, "requiresOpenaiAuth": False}, CodexAuth.NOT_REQUIRED),
        ({"account": None, "requiresOpenaiAuth": True}, CodexAuth.MISSING),
    ],
)
def test_auth_state(result: dict[str, Any], expected: CodexAuth) -> None:
    assert auth_state(result) is expected
