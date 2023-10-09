from datetime import datetime, timezone
from google.cloud import storage
from main import FAST_MODEL, FakeTestErrorException, format_search_results, query_model, RemoteData, Responses, UserReplies, webhook, search_conversations, DATETIME_IN_FORMAT, DATETIME_CONV_FORMAT, message_user, query_embeddings_model, EMBEDDINGS_MODEL
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
assert ADMIN_CHAT_ID is not None, "ADMIN_CHAT_ID environment variable not set"
assert BUCKET is not None, "BUCKET environment variable not set"

TEST_DATETIMES = ["100101020000", "100201010000", "100301010000"]  # my formatting stuff doesn't like dates starting with 0, so don't use those for testing
TEST_DATETIME = TEST_DATETIMES[0]

def example_request_json(msg, from_id=ADMIN_CHAT_ID): 
    return {
    "update_id": 000000000,
    "message": {
        "message_id": 0000,
        "from": {
            "id": from_id,
            "is_bot": False,
            "first_name": "no",
        },
        "chat": {
            "id": from_id,
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
        "date": datetime.strptime(dt, DATETIME_CONV_FORMAT).replace(tzinfo=timezone.utc).timestamp(),
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

def example_search_request_json(query, n_results):
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
        "text": f"/search {n_results} {query}",
    } 
}

@pytest.fixture
def mock_query_model(monkeypatch):
    def _mock_query_model(previous_messages, model="MOCK_MODEL"):
        return "MOCK RESPONSE - test suite ran"
    monkeypatch.setattr("main.query_model", _mock_query_model)

@pytest.fixture
def replace_error(monkeypatch):
    def _replace_error(error_type_name):
        monkeypatch.setattr(error_type_name, FakeTestErrorException)
    return _replace_error

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
def autofilter_search(monkeypatch):
    def _apply(filter):
        old_search = search_conversations
        def new_search(remote_data):
            return old_search(remote_data, filter)
        monkeypatch.setattr("main.search_conversations", new_search)
    return _apply

def setup_and_teardown(desired_datetime, no_warn_rate_limit=False):
    # raise an exception if desired_datetime is not in TEST_DATETIMES, to avoid deleting a real conversation
    assert desired_datetime in TEST_DATETIMES, f"desired_datetime {desired_datetime} not in TEST_DATETIMES. \
        (this is a safety feature to avoid deleting a real conversation)"

    # get the original live conversation
    remote_data = RemoteData(BUCKET)
    live_conversation = remote_data.get_live_conversation(ADMIN_CHAT_ID)

    # if the live conversation matches the desired conversation, and no_warn_rate_limit is True, then don't do anything
    if live_conversation == desired_datetime and no_warn_rate_limit:
        yield  # run the test before no-op teardown
        return
    
    # delete any old conversation with timestamp TEST_DATETIME (this is done initially as well as at the end in case of errors)
    def del_test_conversation():
        test_conversation_path = f"{ADMIN_CHAT_ID}/{desired_datetime}"
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET)
        blob = bucket.blob(test_conversation_path)
        if blob.exists():
            blob.delete()
    del_test_conversation()

    # create a new conversation with timestamp TEST_DATETIME
    resp = webhook(MockRequest(new_conversation_request_json(desired_datetime)))
    assert resp == Responses.OKAY, "setup failed: new conversation request errored"
    # check that the new conversation is the live conversation
    cur_conv = remote_data.get_live_conversation(ADMIN_CHAT_ID)
    assert cur_conv == desired_datetime, f"setup failed: new conversation request did not set the live conversation correctly. live conversation is {cur_conv}, desired conversation is {desired_datetime}"

    # run the test
    yield

    # set the live conversation back to what it was before
    resp = webhook(MockRequest(example_switch_conversation_request_json(live_conversation)))
    assert resp == Responses.OKAY, "teardown failed: switch conversation request errored (original conversation not restored)"
    # check that the live conversation is back to what it was before
    cur_conv = remote_data.get_live_conversation(ADMIN_CHAT_ID)
    assert cur_conv == live_conversation, f"teardown failed: switch conversation request did not set the live conversation correctly (original conversation not restored). live conversation is {cur_conv}, desired conversation is {live_conversation}"

    # delete the test conversation
    del_test_conversation()

@pytest.mark.usefixtures("mock_query_model")
def test_webhook(monkeypatch):
    """Test that the webhook returns a 200 status code. Doesn't send a real query to OpenAI, but does send a real response to the admin."""
    monkeypatch.setattr("main.message_user", message_user)
    
    s = setup_and_teardown(TEST_DATETIME, no_warn_rate_limit=True)
    next(s)
    request = MockRequest(example_request_json("test request"))

    resp = webhook(request)
    assert resp == Responses.OKAY, "webhook request errored"
    try:
        next(s)
    except StopIteration:
        pass

@pytest.fixture(autouse=True)
def mock_message_user(monkeypatch):
    async def _mock_message_user(chat_id, msg, context, parse_mode=None):
        print(f"mock_message_user called with {chat_id}, {msg}, {context}, {parse_mode}")
    monkeypatch.setattr("main.message_user", _mock_message_user)

@pytest.fixture
def mock_message_user_gen(monkeypatch):
    def _apply_mock(filter):
        async def mock_message_user_(chat_id, msg, context, parse_mode=None):
            if filter(msg):
                raise FakeTestErrorException("received key message")
            print(f"mock_message_user_gen called with non-key message in chat {chat_id}, with context {context}, in parse mode {parse_mode}:\n{msg}\n")
        monkeypatch.setattr("main.message_user", mock_message_user_)
    return _apply_mock

@pytest.fixture(autouse=True)
def setup_and_teardown_default(mock_message_user):
    st_obj = setup_and_teardown(TEST_DATETIME, no_warn_rate_limit=True)
    next(st_obj)
    yield
    try:
        next(st_obj)
    except StopIteration:
        pass

def test_query_model():
    """Run a query on gpt-3.5-turbo and make sure it returns a string."""
    assert isinstance(query_model(example_query, FAST_MODEL), str), "query_model did not return a string"

# mock_message_user shouldn't be used, but is here to prevent spamming in case of errors
@pytest.mark.usefixtures("mock_query_model", "mock_unsupported_message")
def test_webhook_image():
    request = MockRequest(example_image_request_json)
    # check that the unsupported message function is called
    resp = webhook(request)
    assert resp == Responses.TEST, "unsupported message function was not called when expected"

@pytest.mark.usefixtures("mock_query_model", "mock_read_document")
def test_webhook_document(mock_message_user_gen):
    mock_message_user_gen(lambda x: x == Responses.DOCUMENT_RECEIVED)
    request = MockRequest(example_document_request_json)

    resp = webhook(request)
    assert resp == Responses.TEST, "document received response was not sent when expected"

def test_rate_limit(replace_error):
    replace_error("main.RateLimitException")
    request = MockRequest(new_conversation_request_json(TEST_DATETIME))

    resp = webhook(request)
    assert resp == Responses.TEST, "rate limit response was not sent when expected"

@pytest.mark.usefixtures("mock_query_model")
def test_search(mock_message_user_gen, autofilter_search):
    autofilter_search(lambda x: x in TEST_DATETIMES)

    # first, make two conversations to search
    c1id = TEST_DATETIMES[1]
    c2id = TEST_DATETIMES[2]
    c1 = setup_and_teardown(c1id)
    c2 = setup_and_teardown(c2id)

    next(c1)
    request = MockRequest(example_request_json("/no_response the HORSE is a glorious animal"))
    resp = webhook(request)
    assert resp == Responses.OKAY, "first conversation errored"

    next(c2)
    request = MockRequest(example_request_json("/no_response how to ride a horse"))
    resp = webhook(request)
    assert resp == Responses.OKAY, "second conversation errored"

    # the results should have that the first message is the first result and the second message is the second result
    def check_if_result_is_close(msg):
        # check whether msg has the format of format_search_results([(c1id, 0.9685448548924412), (c2id, 0.950411131658208)]),
        # but where the values may differ slightly
        eps = 0.001
        refval1 = 0.99384
        refval2 = 0.97358
        ref_msg = format_search_results([(c1id, refval1), (c2id, refval2)])
        # split ref_msg into where the given numbers occur
        ref_msg_split1 = ref_msg.split(str(refval1))
        if not len(ref_msg_split1) == 2:
            print(f"length of ref_msg_split1 is {len(ref_msg_split1)}, not 2; not a match")
            return False
        ref_msg_split2 = ref_msg_split1[1].split(str(refval2))
        if not len(ref_msg_split2) == 2:
            print(f"length of ref_msg_split2 is {len(ref_msg_split2)}, not 2; not a match")
            return False
        ref_msg_split = [ref_msg_split1[0], ref_msg_split2[0], ref_msg_split2[1]]
        # split msg into where the given numbers occur
        msg_split1 = msg[:len(ref_msg_split[0])]
        if not msg_split1 == ref_msg_split[0]:
            print(f"msg_split1 is {msg_split1}, not {ref_msg_split[0]}; not a match")
            return False
        # the following section of msg should be a number (which may not be the same length as refval1).
        # check that it is close to refval1
        def get_prefix_float(s):
            prefix = ""
            decimalpointseen = False
            for c in s:
                if c == ".":
                    if decimalpointseen:
                        break
                    else:
                        decimalpointseen = True
                        prefix += c
                elif c.isdigit():
                    prefix += c
                else:
                    break
            return prefix
        msgval1str = get_prefix_float(msg[len(ref_msg_split[0]):])
        msgval1 = float(msgval1str)
        if not abs(msgval1 - refval1) < eps:
            print(f"msgval1 is {msgval1}, which is more than {eps} away from {refval1}; not a match")
            return False
        offset_after_val1 = len(ref_msg_split[0]) + len(msgval1str)
        if not msg[offset_after_val1:offset_after_val1 + len(ref_msg_split[1])] == ref_msg_split[1]:
            print(f"msg[offset_after_val1:offset_after_val1 + len(ref_msg_split[1])] is {msg[offset_after_val1:offset_after_val1 + len(ref_msg_split[1])]}, not {ref_msg_split[1]}; not a match")
            return False
        # the following section of msg should be a number (which may not be the same length as refval2).
        # check that it is close to refval2
        msgval2str = get_prefix_float(msg[offset_after_val1 + len(ref_msg_split[1]):])
        msgval2 = float(msgval2str)
        if not abs(msgval2 - refval2) < eps:
            print(f"msgval2 is {msgval2}, which is more than {eps} away from {refval2}; not a match")
            return False
        offset_after_val2 = offset_after_val1 + len(ref_msg_split[1]) + len(msgval2str)
        if not msg[offset_after_val2:] == ref_msg_split[2]:
            print(f"msg[offset_after_val2:] is {msg[offset_after_val2:]}, not {ref_msg_split[2]}; not a match")
            return False
        return True
        
    mock_message_user_gen(check_if_result_is_close)

    # now, do the semantic search
    request = MockRequest(example_search_request_json("the HORSE is a noble animal", 2))
    resp = webhook(request)
    assert resp == Responses.TEST, f"search response did not match template"

    try:
       next(c2)
    except StopIteration:
        pass
    try:
        next(c1)
    except StopIteration:
        pass

@pytest.mark.usefixtures("mock_query_model")
def test_long_message_embedding(replace_error, mock_message_user_gen, autofilter_search, monkeypatch):
    old_qem = query_embeddings_model
    def new_qem(previous_messages, model=EMBEDDINGS_MODEL, average_ok=True):
        return old_qem(previous_messages, model, False)
    monkeypatch.setattr("main.query_embeddings_model", new_qem)

    # prepare for the long message
    replace_error("main.TooLongEmbeddingException")

    # create a very long message
    request = MockRequest(example_request_json("a " * 10000))
    resp = webhook(request)
    assert resp == Responses.OKAY, "long message errored"

    # now send a search query so that the previous message is embedded
    request = MockRequest(example_search_request_json("a " * 100, 1))
    resp = webhook(request)
    assert resp == Responses.TEST, "averaging of embeddings failed"

def test_ensure_admin(mock_message_user_gen):
    mock_message_user_gen(lambda x: x == UserReplies.NON_ADMIN)
    request = MockRequest(example_request_json("/usage_stats", from_id=0))
    resp = webhook(request)
    assert resp == Responses.TEST, "ensure_admin did not send the expected response for non-admin user"

    request = MockRequest(example_request_json("/usage_stats", from_id=ADMIN_CHAT_ID))
    resp = webhook(request)
    assert resp == Responses.OKAY, "ensure_admin did not send the expected response for admin user"
    