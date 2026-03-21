from app.telegram.handler import TelegramClient, TelegramMessage, TelegramPhoto, parse_supported_command


def test_parse_addbook_text_command():
    message = TelegramMessage(message_id=1, chat={"id": 123}, text="/addbook Dune")

    command = parse_supported_command(message)

    assert command is not None
    assert command.kind == "text"
    assert command.title == "Dune"


def test_parse_addbook_without_title_returns_help_command():
    message = TelegramMessage(message_id=1, chat={"id": 123}, text="/addbook")

    command = parse_supported_command(message)

    assert command is not None
    assert command.kind == "addbook_help"


def test_parse_scanbook_without_photo_returns_help_command():
    message = TelegramMessage(message_id=1, chat={"id": 123}, text="/scanbook")

    command = parse_supported_command(message)

    assert command is not None
    assert command.kind == "scanbook_help"


def test_parse_scanbook_photo_caption():
    message = TelegramMessage(
        message_id=1,
        chat={"id": 123},
        caption="/scanbook",
        photo=[
            TelegramPhoto(file_id="small", width=100, height=100),
            TelegramPhoto(file_id="large", width=500, height=500),
        ],
    )

    command = parse_supported_command(message)

    assert command is not None
    assert command.kind == "image"
    assert command.file_id == "large"


def test_parse_old_add_book_formats_are_rejected():
    text_message = TelegramMessage(message_id=1, chat={"id": 123}, text="Add Book Dune")
    photo_message = TelegramMessage(
        message_id=2,
        chat={"id": 123},
        caption="Add Book",
        photo=[TelegramPhoto(file_id="img", width=100, height=100)],
    )

    assert parse_supported_command(text_message) is None
    assert parse_supported_command(photo_message) is None


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        return _FakeResponse()


def test_set_my_commands_registers_expected_commands():
    client = TelegramClient("token")
    fake_client = _FakeHttpClient()
    client._client = fake_client  # type: ignore[assignment]

    import asyncio

    asyncio.run(client.set_my_commands())

    assert len(fake_client.calls) == 1
    _, payload = fake_client.calls[0]
    assert payload["commands"] == [
        {"command": "addbook", "description": "Add Book to Reading List"},
        {"command": "scanbook", "description": "Scan a book to add"},
    ]
