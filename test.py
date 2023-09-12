from datetime import datetime
from google.cloud import storage
from main import FAST_MODEL, FakeTestErrorException, format_search_results, query_model, RemoteData, Responses, UserReplies, webhook
from os import getenv
import pytest
from time import time

class MockRequest():
    def __init__(self, json):
        self.json = json
        self.method = "POST"

    def get_json(self, force=False):
        return self.json

ADMIN_CHAT_ID = getenv("ADMIN_CHAT_ID")  # so it doesn't complain about being unverified
BUCKET = getenv("BUCKET")
assert ADMIN_CHAT_ID is not None
assert BUCKET is not None

DATETIME_CONV_FORMAT = "%Y%m%d%H%M"
DATETIME_IN_FORMAT = "%Y-%m-%d-%H-%M-%S"
TEST_DATETIME = "000101020000"

def example_request_json(msg): 
    return {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "from": {
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            "first_name": "no",
        },
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
        },
        "date": 0000000000,
        "text": msg,
    } 
}

example_image_request_json = {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
            },
        "date": 0000000000,
        "from": {
            "first_name": "no",
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            },
        "photo": [
            {
                "file_id": "blahblahblah",
                "file_unique_id": "blah",
                "file_size": 1920,
                "width": 90,
                "height": 80,
            },
        ]
    }  
}

example_document_request_json = {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
            },
        "date": 0000000000,
        "from": {
            "first_name": "no",
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            },
        "document": {
            "file_id": "blahblahblah",
            "file_unique_id": "blah",
            "file_name": "test.txt",
            "mime_type": "text/plain",
            "file_size": 1920,
        }
    }  
}

def new_conversation_request_json(dt):
    return {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "from": {
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            "first_name": "no",
        },
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
        },
        "date": datetime.strptime(dt, DATETIME_CONV_FORMAT).timestamp(),
        "text": f"/new_conversation",
    } 
}

def example_switch_conversation_request_json(dt):
    return {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "from": {
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            "first_name": "no",
        },
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
        },
        "date": time(),
        "text": f"/switch_conversation {datetime.strptime(dt, DATETIME_CONV_FORMAT).strftime(DATETIME_IN_FORMAT)}",
    } 
}

example_query = [{"role": "system", "content": "say 'h'"}]

example_search_request_json = {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "from": {
            "id": ADMIN_CHAT_ID,
            "is_bot": False,
            "first_name": "no",
        },
        "chat": {
            "id": ADMIN_CHAT_ID,
            "first_name": "no",
            "type": "private",
        },
        "date": 0000000000,
        "text": "/search 2 the HORSE is a noble animal",
    } 
}

@pytest.fixture
def mock_query_model(monkeypatch):
    def mock_query_model(previous_messages, model="MOCK_MODEL"):
        return "MOCK RESPONSE"
    monkeypatch.setattr("main.query_model", mock_query_model)

@pytest.fixture
def mock_message_user(monkeypatch):
    async def mock_message_user(chat_id, msg, context, parse_mode=None):
        print(f"mock_message_user called with {chat_id}, {msg}, {context}, {parse_mode}")
    monkeypatch.setattr("main.message_user", mock_message_user)

# TODO remove
@pytest.fixture
def mock_message_user_q(monkeypatch):
    async def mock_message_user(chat_id, msg, context, parse_mode=None):

        print("QQQ LIVE CONV", RemoteData(BUCKET).get_live_conversation(ADMIN_CHAT_ID))
        print(f"QQQQQQ called with {chat_id}, {msg}, {context}, {parse_mode}")
    monkeypatch.setattr("main.message_user", mock_message_user)

@pytest.fixture
def mock_message_user_gen(monkeypatch):
    def _apply_mock(key_message):
        async def mock_message_user_(chat_id, msg, context, parse_mode=None):
            if msg == key_message:
                raise FakeTestErrorException("received fake message")
            print(f"mock_message_user called with {chat_id}, {msg}, {context}, {parse_mode}")
        monkeypatch.setattr("main.message_user", mock_message_user_)
    return _apply_mock

@pytest.fixture
def mock_unsupported_message(monkeypatch):
    async def mock_unsupported_message(update, context):
        raise FakeTestErrorException("unsupported_message")
    monkeypatch.setattr("main.unsupported_message", mock_unsupported_message)

@pytest.fixture
def mock_read_document(monkeypatch):
    async def mock_read_document(message):
        return "FAKE DOCUMENT"
    monkeypatch.setattr("main.read_document", mock_read_document)

@pytest.fixture
def mock_message_user_gen(monkeypatch):
    def _apply_mock(key_message):
        async def mock_message_user_(chat_id, msg, context, parse_mode=None):
            if msg == key_message:
                raise FakeTestErrorException("received fake message")
            print(f"mock_message_user called with {chat_id}, {msg}, {context}, {parse_mode}")
        monkeypatch.setattr("main.message_user", mock_message_user_)
    return _apply_mock

def setup_and_teardown(desired_datetime):
    # delete any old conversation with timestamp TEST_DATETIME (this is done initially as well as at the end in case of errors)
    def del_test_conversation():
        test_conversation_path = f"{BUCKET}/{ADMIN_CHAT_ID}/{desired_datetime}"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET)
        blob = bucket.blob(test_conversation_path)
        if blob.exists():
            blob.delete()
    del_test_conversation()

    # get the original live conversation
    remote_data = RemoteData(BUCKET)
    live_conversation = remote_data.get_live_conversation(ADMIN_CHAT_ID)

    # create a new conversation with timestamp TEST_DATETIME
    resp = webhook(MockRequest(new_conversation_request_json(desired_datetime)))
    assert resp == Responses.OKAY

    # run the test
    yield

    # set the live conversation back to what it was before
    resp = webhook(MockRequest(example_switch_conversation_request_json(live_conversation)))
    assert resp == Responses.OKAY

    # delete the test conversation
    del_test_conversation()

@pytest.fixture(autouse=True)
def setup_and_teardown_default():
    setup_and_teardown(TEST_DATETIME)

@pytest.mark.usefixtures("mock_query_model")
def test_webhook():
    """Test that the webhook returns a 200 status code. Doesn't send a real query to OpenAI, but does send a real response to the admin."""
    request = MockRequest(example_request_json("test request"))

    resp = webhook(request)
    assert resp == Responses.OKAY

def test_query_model():
    """Run a query on gpt-3.5-turbo and make sure it returns a string."""
    assert isinstance(query_model(example_query, FAST_MODEL), str)

# mock_message_user shouldn't be used, but is here to prevent spamming in case of errors
@pytest.mark.usefixtures("mock_query_model", "mock_message_user", "mock_unsupported_message")
def test_webhook_image():
    request = MockRequest(example_image_request_json)
    # check that the unsupported message function is called
    resp = webhook(request)
    assert resp == Responses.TEST

@pytest.mark.usefixtures("mock_query_model", "mock_read_document")
def test_webhook_document(mock_message_user_gen):
    mock_message_user_gen(Responses.DOCUMENT_RECEIVED)
    request = MockRequest(example_document_request_json)

    resp = webhook(request)
    assert resp == Responses.TEST

@pytest.mark.usefixtures("mock_message_user_q")  # TODO remove and put back
def test_rate_limit(mock_message_user_gen):
    # mock_message_user_gen(UserReplies.RATE_LIMIT)
    request = MockRequest(new_conversation_request_json(TEST_DATETIME))

    resp = webhook(request)
    assert resp == Responses.TEST

@pytest.mark.usefixtures("mock_query_model", "mock_message_user")
def test_search(mock_message_user_gen):

    # first, make two conversations to search
    c1id = "000201010000"
    c2id = "000301010000"
    c1 = setup_and_teardown(c1id)
    c2 = setup_and_teardown(c2id)

    next(c1)
    request = MockRequest(example_request_json("the HORSE is a glorious animal"))
    resp = webhook(request)
    assert resp == Responses.OKAY

    next(c2)
    request = MockRequest(example_request_json("how to ride a horse"))
    resp = webhook(request)
    assert resp == Responses.OKAY

    # the results should have that the first message is the first result and the second message is the second result
    mock_message_user_gen(format_search_results([(c1id, 0.9732164302232387), (c2id, 0.8288876602593767)]))

    # now, do the semantic search
    request = MockRequest(example_search_request_json)
    resp = webhook(request)
    assert resp == Responses.TEST

    next(c2)
    next(c1)
