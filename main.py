import logging
import openai

from asyncio import run as asyncio_run
from datetime import datetime, timezone
from google.cloud import storage
from json import dumps, JSONDecodeError, load, loads
# from openai.embeddings_utils import get_embeddings, cosine_similarity
from os import getenv, path, remove
from random import choices as rand_choices
from string import ascii_letters, digits
from telegram import Message, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CallbackContext, ContextTypes, CommandHandler, ExtBot, MessageHandler, filters
from tiktoken import encoding_for_model
from time import time
from typing import List, Optional, Tuple


# === Setup ===========================================================================================================

MODEL = "gpt-4"  # "text-davinci-003"
LONG_MODEL = "gpt-4-32k"
FAST_MODEL = "gpt-3.5-turbo"
LONG_FAST_MODEL = "gpt-3.5-turbo-16k"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

MODEL_DATA = {
    "gpt-4": {
        "max_tokens": 8192,
    },
    "gpt-4-32k": {
        "max_tokens": 32768,
    },
    "gpt-3.5-turbo": {
        "max_tokens": 4097,
    },
    "gpt-3.5-turbo-16k" : {
        "max_tokens": 16385,
    },
    "text-embedding-ada-002": {
        "max_tokens": 8191,
    },
}

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful Telegram chatbot. You can use Telegram markdown message formatting, e.g. `inline code`, ```c++\ncode written in c++```, *bold*, and _italic_."}

HELP_MESSAGE = "i'm a chatbot for interfacing with openai's api.\n\nby default, i respond to direct messages with `gpt-4`; you can also use the /turbo command to send a message to `gpt-3.5-turbo` instead (or, if your message is sufficiently long, `gpt-3.5-turbo-16k`).\n\ni can also read documents, which you can send to me as attachments. i'll respond to the document if it has a caption, or just acknowledge receipt if it doesn't.\n\ni remember your messages until you send me a /new\_conversation command, at which point i'll forget everything you've said before."

class Message:
    def __init__(self, message_id, message_text):
        self.message_id = message_id
        self.message_text = message_text

class Responses():
    # has a class attribute for each of the possible outputs that go along with a status code    
    OKAY = ("okay", 200)
    ERROR = ("error", 400)
    TEST = ("test", 999)  # response paired with a code other than 200 (i)
    NO_STATUS = ("no_status", 200)
    UNVERIFIED = ("unverified", 200)
    UNSUPPORTED = ("unsupported", 200)
    RATE_LIMIT = ("rate_limit", 200)
    INTERNAL_ERROR = ("internal_error", 200)
    NONE_UPDATE = ("none_update", 200)
    META_ERROR = ("meta_error", 200)

    DOCUMENT_RECEIVED = "document_received"

class UserReplies():
    UNSUPPORTED = "sorry, i can only read documents"
    RATE_LIMIT = "slow down there!"

# https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
    ]
)


# == Google Cloud stuff ================================================================================================

class UnverifiedException(Exception):
    code = Responses.UNVERIFIED

class UnsupportedMessageException(Exception):
    code = Responses.UNSUPPORTED

class RateLimitException(Exception):
    code = Responses.RATE_LIMIT

class FakeTestErrorException(Exception):
    code = Responses.TEST

class InvalidJSONException(Exception):
    pass

# the RemoteData class is for interfacing with the bucket containing user data.
# the bucket will contain a folder for each chat the bot is in;
# each folder will contain a file for each conversation belonging to that chat.
class RemoteData:
    def __init__(self, bucket):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.get_bucket(bucket)

    def write_conversation(self, chat_id, conversation_data):
        destination_blob_name = f"{chat_id}/{self.get_live_conversation(chat_id)}"
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(dumps(conversation_data))
        logging.info('File {} uploaded'.format(destination_blob_name))
        
    def read_live_conversation(self, chat_id):
        return self.read_conversation(chat_id, self.get_live_conversation(chat_id))
    
    def read_conversation(self, chat_id, conv_id):
        source_blob_name = f"{chat_id}/{conv_id}"
        blob = self.bucket.blob(source_blob_name)
        logging.info('File {} downloaded'.format(source_blob_name))
        return loads(blob.download_as_bytes())
    
    def append_conversation(self, chat_id, role, content):
        conversation_data = self.read_live_conversation(chat_id)
        conversation_data.append({"role": role, "content": content})
        self.write_conversation(chat_id, conversation_data)
        logging.info(f"appended to live conversation for chat {chat_id}")
        # delete the corresponding embedding since it has changed
        self.delete_embedding(chat_id)

    def start_new_conversation(self, update: Update):
        chat_id = update.effective_chat.id
        # get the time the message was sent, converted to utc
        sent_time = update.message.date.astimezone(tz=timezone.utc)
        new_id = sent_time.strftime("%Y%m%d%H%M")
        # check if the live conversation is the same as the new conversation
        if new_id == self.get_live_conversation(chat_id):
            raise RateLimitException()
        self.set_live_conversation(chat_id, new_id)
        self.write_conversation(chat_id, [SYSTEM_PROMPT])
        logging.info(f"created new conversation for chat {chat_id} with id {new_id}")
        return new_id
    
    def set_verified(self, chat_id, val):
        destination_blob_name = f"{chat_id}/info/verified"
        blob = self.bucket.blob(destination_blob_name)
        if val:
            # create an empty blob
            blob.upload_from_string("")
        else:
            # delete the blob
            blob.delete()
    
    def is_verified(self, chat_id):
        destination_blob_name = f"{chat_id}/info/verified"
        blob = self.bucket.blob(destination_blob_name)
        return blob.exists()
    
    def set_live_conversation(self, chat_id, conversation_id: str):
        # check if the chat_id has been verified.
        # if public access is not allowed, raise an exception if it does not (user has not been verified yet)
        if getenv("ALLOW_PUBLIC") != "true":
            if not self.is_verified(chat_id):
                logging.info(f"unverified chat {chat_id} tried to send a message")
                raise UnverifiedException(chat_id)

        destination_blob_name = f"{chat_id}/info/live"
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(conversation_id)
        logging.info(f"set live conversation for chat {chat_id} to {conversation_id}")

    def get_live_conversation(self, chat_id) -> Optional[str]:
        """Returns the id of the conversation that is currently live for the given chat, or None if there is no live conversation."""
        source_blob_name = f"{chat_id}/info/live"
        blob = self.bucket.blob(source_blob_name)
        if blob.exists():
            live_conversation_id_bin = blob.download_as_bytes()
            # convert from bytes to string
            live_conversation_id = live_conversation_id_bin.decode("utf-8")
            logging.info(f"live conversation for chat {chat_id} is {live_conversation_id}")
            return live_conversation_id
        else:
            logging.info(f"no live conversation for chat {chat_id}")
            return None
        
    def create_embedding(self, chat_id, conv_id):
        logging.info(f"creating embedding for conversation {chat_id}/{conv_id}")
        content = self.read_conversation(chat_id, conv_id)
        try:
            query = dumps(content)
        except JSONDecodeError:
            raise InvalidJSONException()
        embedding_text = dumps(query_embeddings_model(query, model=EMBEDDINGS_MODEL))
        destination_blob_name = f"{embeddings_info_path(chat_id)}/{conv_id}"
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(embedding_text)
    
    def delete_embedding(self, chat_id):
        logging.info(f"deleting embedding for chat {chat_id}")
        live_conversation_id = self.get_live_conversation(chat_id)
        source_blob_name = f"{embeddings_info_path(chat_id)}/{live_conversation_id}"
        blob = self.bucket.blob(source_blob_name)
        if blob.exists():
            blob.delete()
        
def embeddings_info_path(chat_id):
    return f"{chat_id}/info/embeddings"

# === Telegram stuff ===================================================================================================

# for some reason a real CallbackContext is not working with the webhook setup
class FakeContext(CallbackContext):
    def __init__(self, bot, message, err):
        self._bot = bot
        self._error = err

        # if message starts with '/', it's a command
        if message.text is not None and message.text.startswith("/"):
            self._args = message.text.split(" ", 1)[1:]
        else:
            self._args = []
    
    @property
    def bot(self):
        return self._bot
    
    @property
    def args(self):
        return self._args
    
    @property
    def error(self):
        return self._error
    
    def set_error(self, err):
        self._error = err

async def message_user(chat_id: int, msg: str, context: ContextTypes.DEFAULT_TYPE, parse_mode=ParseMode.MARKDOWN):
    logging.debug(f"sending message to chat {chat_id}: {msg}")

    try:
        if parse_mode is not None:
            await context.bot.send_message(chat_id=chat_id,
                                           text=msg,
                                           parse_mode=parse_mode)
            logging.debug(f"sent message to chat {chat_id}: {msg}")
            return
    except BadRequest as e:
        logging.error(f"bad request: {e}. trying again with parse_mode=None")

    await context.bot.send_message(chat_id=chat_id,
                                   text=msg)
    logging.debug(f"sent plaintext message to chat {chat_id}: {msg}")

async def admin_message(msg: str, context: ContextTypes.DEFAULT_TYPE):
    logging.debug(f"sending admin message: {msg}")
    await message_user(int(getenv("ADMIN_CHAT_ID")), msg, context)

def start(_):
    async def start_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.debug("adding new chat")
        await message_user(update.effective_chat.id, "welcome! please wait to be verified. (a human has to do it, so it might be slow)", context)

        # message the admin
        await admin_message(f"new chat!\nuser info: {update.effective_chat}\nchat id:", context)
        await admin_message(f"{update.effective_chat.id}", context)
    return start_

def ensure_admin(_):
    async def ensure_admin_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # check if the sender is the admin
        if update.effective_chat.id != int(getenv("ADMIN_CHAT_ID")):
            logging.info(f"unauthorized user {update.effective_chat.id} tried to send a message")
            await message_user(update.effective_chat.id, "nice try", context)
            return False
        return True
    return ensure_admin_

def verify(remote_data: RemoteData):
    # verify the chat with the given id
    async def verify_(update: Update, context: ContextTypes.DEFAULT_TYPE):        
        # check that the message is a chat id
        async def admin_message_err():
            await message_user(update.effective_chat.id, "this doesn't look like a chat id to me :/", context)

        if len(context.args) != 1:
            logging.info(f"admin {update.effective_chat.id} tried to verify a chat, but didn't give a chat id")
            await admin_message_err()
            return
        try:
            chat_id = int(context.args[0])
        except ValueError:
            logging.info(f"admin {update.effective_chat.id} tried to verify a chat, but gave an invalid chat id")
            await admin_message_err()
            return
        
        # verify the chat
        try:
            await message_user(chat_id, "you've been verified! you can now send messages to the bot", context)
        except Exception as e:
            logging.info(f"admin {update.effective_chat.id} tried to verify a chat, but the chat id was invalid: exception {e}")
            await message_user(update.effective_chat.id, f"i couldn't send a message to chat {chat_id}", context)
            return

        remote_data.set_verified(chat_id, True)
        logging.info(f"verified chat {chat_id}")
        await admin_message(f"verified chat {chat_id}", context)
        
    return verify_

# TODO: add "unverify" command

def usage_stats(remote_data: RemoteData):
    async def usage_stats_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        metrics = {}
        # get the total number of messages from each user
        # loop through top-level directories
        # list top-level directories, using delimiter and prefix to emulate hierarchical structure

        for blob in remote_data.bucket.list_blobs():
            # parse this into a chat ID and timestamp; if it doesn't match that pattern (i.e. it is the source code zipfile, an 'info/live' file, etc), skip it
            split_path = blob.name.split("/")
            if len(split_path) != 2:
                continue
            chat, timestamp = split_path
            if not timestamp.isdigit():
                continue
            metrics[chat] = metrics.get(chat, 0) + 1
        
        # format the metrics
        metrics_str = "\n".join([f"{chat_id}: {num_msgs}" for chat_id, num_msgs in metrics.items()])

        logging.info(f"usage stats: {metrics_str}")

        await message_user(update.effective_chat.id, metrics_str, context)
    return usage_stats_
    
def help(_):
    async def help_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.debug("sending help message")
        await message_user(update.effective_chat.id, HELP_MESSAGE, context)

def new_conversation(remote_data: RemoteData):
    async def new_conversation_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        remote_data.start_new_conversation(update)
        logging.debug("new conversation started")
        await message_user(update.effective_chat.id, "new conversation started", context)
    return new_conversation_

async def record_user_message(remote_data: RemoteData, update: Update, process_msg=lambda msg: msg.text):
    current_chat_id = update.effective_chat.id
    live_conversation_id = remote_data.get_live_conversation(current_chat_id)
    if live_conversation_id is None:
            live_conversation_id = remote_data.start_new_conversation(update)
    remote_data.append_conversation(current_chat_id, "user", process_msg(update.message))
    logging.info(f"recorded message from chat {current_chat_id} with id {live_conversation_id}")

def count_tokens(text: str, model: str):
    enc = encoding_for_model(model)
    return len(enc.encode(text))

def get_n_tokens(text: str, model: str, n: int):
    enc = encoding_for_model(model)
    tokenized_text = enc.encode(text)[:n]
    return enc.decode(tokenized_text)

async def respond_conversation(remote_data: RemoteData, current_chat_id: int, context: ContextTypes.DEFAULT_TYPE, model=MODEL):
    chat_history = remote_data.read_live_conversation(current_chat_id)

    # use the long context model if necessary; error if that's not sufficient
    max_tokens = MODEL_DATA[model]["max_tokens"]

    # for the ChatCompletion models, this isn't the exact number of tokens, but it's kind of unclear how to actually get that value.
    # this should at least err in the direction of overcounting tokens, which is ok since we need some space for a response.
    token_num = count_tokens(dumps(chat_history), model)
    if token_num > max_tokens:
        logging.info(f"chat history is {token_num} tokens, which is more than the {max_tokens} token limit; trying long context model")
        if model == MODEL:
            model = LONG_MODEL
        elif model == FAST_MODEL:
            model = LONG_FAST_MODEL
        else:
            logging.error("invalid model")
            raise Exception("invalid model")
        # check that the long context model is sufficient
        max_tokens = MODEL_DATA[model]["max_tokens"]
        if token_num > max_tokens:
            logging.error(f"chat history is {token_num} tokens, which is more than the {max_tokens} token limit; giving up")
            await message_user(current_chat_id, f"chat history is {token_num} tokens, which is more than the {max_tokens} token limit", context, parse_mode=None)
            return
    
    # query the model
    logging.info(f"querying model {model}...")
    try:
        response = query_model(chat_history, model=model)
    except Exception as e:
        # TODO: should maybe give some option to remove the last response so it doesn't ruin the whole conversaion,
        #   if that turns out to be a problem
        logging.error(e)
        await message_user(current_chat_id, f"model failed with error: {e}", context, parse_mode=None)
        return
    
    # store the model's response
    remote_data.append_conversation(current_chat_id, "assistant", response)

    # reply to the user
    await message_user(current_chat_id, response, context)

async def regular_message_core(remote_data: RemoteData, update: Update, context: ContextTypes.DEFAULT_TYPE, process_msg=lambda msg: msg.text, model=MODEL):
    logging.info(f"update received at {time()}: {update}")
    
    await record_user_message(remote_data, update, process_msg)
    current_chat_id = update.effective_chat.id
    await respond_conversation(remote_data, current_chat_id, context, model)

def regular_message(remote_data: RemoteData):
    async def regular_message_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await regular_message_core(remote_data, update, context)
    return regular_message_

def is_turbo(text: str):
    return text.startswith(f"/{Commands.TURBO}")

def remove_turbo(text: str):
    return text[len(Commands.TURBO) + 2:]

def turbo_message(remote_data: RemoteData):
    async def turbo_message_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info("turbo message received")
        await regular_message_core(remote_data, update, context, process_msg=lambda msg: remove_turbo(msg.text), model=FAST_MODEL)
    return turbo_message_

def no_response(remote_data: RemoteData):
    async def no_response_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info(f"message with no response requested received at {time()}: {update}")
        def removenoresponsecommand(msg):
            return msg.text[len(Commands.NO_RESPONSE) + 2:]

        await record_user_message(remote_data, update, process_msg=removenoresponsecommand)
    return no_response_

async def record_document(remote_data: RemoteData, update: Update):
    content = await read_document(update.message)
    await record_user_message(remote_data, update, process_msg=lambda msg: format_document(msg.document.file_name, content, msg.caption))

def handle_document(remote_data: RemoteData):
    async def handle_document_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info(f"document received at {time()}: {update}")
        await record_document(remote_data, update)
        
        current_chat_id = update.effective_chat.id

        # if the document has a caption, respond to the document; otherwise, just acknowledge receipt
        if update.message.caption is not None:
            logging.info("document with caption received")
            if is_turbo(update.message.caption):
                logging.info("responding to turbo document")
                model = FAST_MODEL
            else:
                model = MODEL
            await respond_conversation(remote_data, current_chat_id, context, model=model)
        else:
            logging.info("document without caption received")
            await message_user(current_chat_id, Responses.DOCUMENT_RECEIVED, context)
    return handle_document_

async def read_document(msg: Message):
    # check that the effective attachment is a file rather than a photo, poll, etc
    
    logging.debug(f"msg.effective_attachment: {msg.effective_attachment}")
    if msg.document is None:
        raise UnsupportedMessageException(msg)
    
    new_file = await msg.effective_attachment.get_file()

    # save the file locally to the tmp directory with a random filename
    random_string = ''.join(rand_choices(ascii_letters + digits, k=10))
    local_path = path.join("/tmp", random_string)
    
    await new_file.download_to_drive(custom_path=local_path)

    # read in the file as text
    with open(local_path, 'r') as document:
        content = document.read()

    # delete the file
    remove(local_path)

    # return the content
    return content


def format_document(file_name: str, content: str, caption: Optional[str]):
    content = f"Document title: {file_name}\nDocument content: {content}"
    if caption is not None:
            if is_turbo(caption):
                logging.info("turbo document")
                content += f"\nDocument caption: {remove_turbo(caption)}"
            else:
                content += f"\nDocument caption: {caption}"
    return content

def switch_conversation(remote_data: RemoteData):
    usage_message = f"please specify the UTC date and time you sent the /{Commands.NEW_CONVERSATION} command \
for the desired conversation in YYYY-MM-DD-HH-MM format, \
up to whatever specificity is needed to uniquely identify the conversation \
(e.g. if you may send just 2023-01-31 if you only had one conversation that day)"

    async def switch_conversation_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        async def invalid_message():
            await message_user(chat_id, usage_message, context)
            return
        logging.info(f"chat {chat_id} is trying to switch conversations: {update}")
        # parse the date in YYYY-mm-DD-HH-MM format from the message
        if len(context.args) != 1:
            logging.info(f"chat {chat_id} tried to switch conversations, but provided a number of arguments not equal to 1")
            return await invalid_message()
        split_date = context.args[0].split("-")
        # ensure that there are 1-5 elements
        if len(split_date) < 1 or len(split_date) > 5:
            logging.info(f"chat {chat_id} tried to switch conversations, but provided a date with the wrong number of elements")
            return await invalid_message()
        # ensure that all elements are integers, and the the first has length 4 while the others have length 2
        if not all([x.isdigit() for x in split_date]):
            logging.info(f"chat {chat_id} tried to switch conversations, but provided a date with non-integer elements")
            return await invalid_message()
        if len(split_date[0]) != 4 or not all([len(x) == 2 for x in split_date[1:]]):
            logging.info(f"chat {chat_id} tried to switch conversations, but provided a date with elements of the wrong length")
            return await invalid_message()
        # convert the provided date to datetime object
        try:
            # datetime requires a month and day, so set those to 1 if they are not provided.
            # this will not affect the lookup because the prefix will be taken based on which info is provided
            if len(split_date) < 3:
                processed_date = split_date + ["01"] * (3 - len(split_date))
            else:
                processed_date = split_date
            conv_date = datetime(*[int(x) for x in processed_date])
        except ValueError as e:
            logging.info(f"chat {chat_id} tried to switch conversations, but conversion to a datetime failed: {e}")
            return await invalid_message()
        # get the conversation id prefix for the given date
        conv_id = conv_date.strftime("".join(["%Y","%m","%d","%H","%M"][:len(split_date)]))
        # determine if this uniquely identifies a conversation
        conv_id_prefix = f"{chat_id}/{conv_id}"
        blobs = list(remote_data.bucket.list_blobs(prefix=conv_id_prefix))
        if len(blobs) == 0:
            logging.info(f"chat {chat_id} tried to switch conversations, but there are no conversations for datetime {conv_date}")
            await message_user(chat_id, f"no conversations for given time", context)
            return
        elif len(blobs) > 1:
            logging.info(f"chat {chat_id} tried to switch conversations, but there are multiple conversations for datetime {conv_date}")
            await message_user(chat_id, f"multiple conversations for given time: please specify with more precision, \
e.g. YYYY-MM-DD-HH-MM if there were multiple conversations in one hour", context)
            return
        # set the live conversation to the identified conversation id
        full_conv_id = blobs[0].name.split("/")[-1]
        remote_data.set_live_conversation(chat_id, full_conv_id)
        logging.info(f"chat {chat_id} switched to conversation {full_conv_id}")
        await message_user(chat_id, "conversation switched", context)
    return switch_conversation_

def format_search_results(top_results: List[Tuple[str, float]]):
    """format the results, converting the conversation ids to datetimes"""
    formatted_results = [(datetime.strptime(conv_id, "%Y%m%d%H%M"), sim) for conv_id, sim in top_results]
    return "\n".join([f"{conv_id}: {sim}" for conv_id, sim in formatted_results])

def search_conversations(remote_data: RemoteData):
    # finds the user's top n conversations with the closest embedding to the given query
    async def search_conversations_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info(f"chat {update.effective_chat.id} is trying to search conversations: {context.args}")
        # parse the number of conversations to return
        if len(context.args) < 1:
            logging.info(f"chat {update.effective_chat.id} tried to search conversations, but didn't provide a query")
            return await message_user(update.effective_chat.id, "please provide a desired number of results and search query", context)
        c00 = context.args[0].split(" ")[0]
        if not c00.isdigit():
            logging.info(f"chat {update.effective_chat.id} tried to search conversations, but provided an invalid number of conversations")
            return await message_user(update.effective_chat.id, "please specify the number of conversations to return", context)
        requested_num_results = int(c00)
        # find the rest of the message, cutting off the command and number of results requested
        content_index = update.message.text.find(c00) + len(c00) + 1
        query = update.message.text[content_index:]
        query_embedding = query_embeddings_model(dumps({"role": "user", "content": query}), model=EMBEDDINGS_MODEL)
        # for each conversation, read or create the embedding and compute the similarity
        results = []
        for blob in remote_data.bucket.list_blobs(prefix=f"{update.effective_chat.id}"):
            # check that this is a conversation file (has format chat_id/conversation_id, where both are integers)
            split_path = blob.name.split("/")
            if len(split_path) != 2 or not all([x.isdigit() for x in split_path]):
                continue
            # get the conversation id
            conv_id = split_path[1]
            # check if an embedding has already been created at chat_id/info/embeddings/conversation_id
            embedding_blob = remote_data.bucket.blob(f"{embeddings_info_path(update.effective_chat.id)}/{conv_id}")
            if not embedding_blob.exists():
                # if the embedding doesn't exist, create it
                try:
                    remote_data.create_embedding(update.effective_chat.id, conv_id)
                    assert embedding_blob.exists(), "embedding blob does not exist after creation"
                except InvalidJSONException:
                    continue
            # read the embedding
            embedding = loads(embedding_blob.download_as_bytes())
            # compute the similarity
            sim = cosine_similarity(query_embedding, embedding)
            # add the result to the list
            results.append((conv_id, sim))
        # get the top n results
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:requested_num_results]
        formatted_results_str = format_search_results(top_results)
        logging.info(f"chat {update.effective_chat.id} searched conversations and got results: {formatted_results_str}")
        await message_user(update.effective_chat.id, formatted_results_str, context)

    return search_conversations_

async def invalid_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"invalid command received at {time()}: {update}")
    await message_user(update.effective_chat.id, "sorry, i don't know that command", context)

async def unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"unsupported message received at {time()}: {update}")
    await message_user(update.effective_chat.id, "sorry, i don't know how to handle that type of message", context)

async def error_handler(update: Optional[object], context: CallbackContext):
    """Log the error and send a telegram message to notify the user. Set the status to 'okay' for expected errors, 'error' otherwise."""
    try:
        logging.error(msg="Exception while handling an update:", exc_info=context.error)
        if update is not None:
            # Check if it was an UnverifiedException
            if isinstance(context.error, UnverifiedException):
                logging.info(f"unverified chat {update.effective_chat.id} tried to send a message")
                await message_user(update.effective_chat.id, f"please send /{Commands.START} to get verified", context)
                context.status = 'unverified'
            elif isinstance(context.error, UnsupportedMessageException):
                logging.info("message is not a document")
                await message_user(update.effective_chat.id, UserReplies.UNSUPPORTED, context)
                context.status = 'unsupported'
            elif isinstance(context.error, RateLimitException):
                logging.info("new conversation rate limit exceeded")
                await message_user(update.effective_chat.id, UserReplies.RATE_LIMIT, context)
                context.status = 'rate_limit'
            else:
                await message_user(update.effective_chat.id, f"An error occurred: {context.error}", context)
                context.status = 'internal_error'
        else:
            logging.error("update is None")
            context.status = 'none_update'
    except Exception as e:
        logging.error(f"error sending message to user: {e}")
        context.status = 'meta_error'


# === OpenAI stuff =====================================================================================================

def query_model(previous_messages, model=MODEL):
    response = openai.ChatCompletion.create(model=model, messages=previous_messages)
    logging.info(f"got response: {response}")
    return response['choices'][0]['message']['content']

def query_embeddings_model(previous_messages, model=EMBEDDINGS_MODEL):
    max_tokens = MODEL_DATA[model]["max_tokens"]
    stringified_messages = dumps(previous_messages)
    token_num = count_tokens(stringified_messages, model)
    if token_num <= MODEL_DATA[model]["max_tokens"]:
        texts_to_get = [(stringified_messages, token_num)]
    else:
        texts_to_get = []
        # split the messages into chunks of no more than max_tokens tokens, cleaving at message boundaries if possible
        
        # find the longest prefix of messages that, when stringified, is less than max_tokens tokens
        remaining_messages = previous_messages
        while remaining_messages:
            num_messages_this_cluster = 0
            current_num_tokens = 0
            for i, msg in enumerate(remaining_messages):
                # count the tokens in this message
                msg_n_tokens = count_tokens(dumps(msg), model)
                # if adding this message would put us over the limit (with some slight allowance), stop
                if current_num_tokens + msg_n_tokens > max_tokens - 4*num_messages_this_cluster:
                    break
                # otherwise, add the message to the cluster
                current_num_tokens += msg_n_tokens
                num_messages_this_cluster += 1
            # verify that the cluster, tokenized as a single message, is less than max_tokens tokens.
            # i'm pretty sure this is always true because splitting things should only increase the number of tokens,
            # (and the allowance of 4 extra tokens should account for the extra newlines that might be added in the dumps of the combination)
            # but i want to know if it's not
            new_text = dumps(remaining_messages[:num_messages_this_cluster])
            new_text_num_tokens = count_tokens(new_text, model)
            assert new_text_num_tokens <= max_tokens
            # if we didn't add any messages to the cluster, then the first message is too long to fit in max_tokens tokens
            if num_messages_this_cluster > 0:
                texts_to_get.append((new_text, new_text_num_tokens))
                remaining_messages = remaining_messages[num_messages_this_cluster:]
            else:
                logging.info(f"splitting long message")
                message_to_split = remaining_messages[0]
                # split the message into chunks of at most max_tokens tokens, formatted as separate messages with the same role
                mts_currentcontent = message_to_split.copy()
                mts_currentcontent["content"] = ""
                allowance = 4 + count_tokens(dumps(mts_currentcontent), model)
                remaining_content = message_to_split["content"]
                while remaining_content:
                    content_section = get_n_tokens(remaining_content, model, max_tokens - allowance)
                    assert content_section  # it would be super weird if this were empty (would imply the allowance is way too large or max_tokens is way too small)
                    mts_currentcontent["content"] = content_section     
                    new_text = dumps(mts_currentcontent)
                    new_text_num_tokens = count_tokens(new_text, model)
                    assert new_text_num_tokens <= max_tokens
                    texts_to_get.append((new_text, new_text_num_tokens))
                    remaining_content = remaining_content[len(content_section):]
                remaining_messages = remaining_messages[1:]
        logging.info(f"split messages into {len(texts_to_get)} chunks")
    
    embeddings = get_embeddings([txt for txt, _ in texts_to_get], model='text-embedding-ada-002')
    logging.info(f"got embeddings response: {embeddings}")
    
    # average the embeddings, weighted by the number of tokens in each
    weighted_embeddings = [emb * n_tokens for emb, (_, n_tokens) in zip(embeddings, texts_to_get)]
    average_embedding = sum(weighted_embeddings) / sum([n_tokens for _, n_tokens in texts_to_get])

    return average_embedding    

# you can use openai.embeddings_utils.get_embeddings for this, but it pulls in a ton of dependencies
def get_embeddings(txts, model='text-embedding-ada-002'):
   return openai.Embedding.create(input=txts, model=model)['data'][0]['embedding']

# you can use openai.embeddings_utils.cosine_similarity for this, but it pulls in a ton of dependencies
def cosine_similarity(emb1: List[float], emb2: List[float]):
    # assert that emb1 and emb2 are both lists of floats of the same length
    assert isinstance(emb1, list) and isinstance(emb2, list)
    assert len(emb1) == len(emb2)
    assert all([isinstance(v, float) for v in emb1]) and all([isinstance(v, float) for v in emb2])
    def sq_norm(vs):
        return sum([pow(v, 2) for v in vs])
    emb1_norm = pow(sq_norm(emb1), 0.5)
    emb1_normed = [v/emb1_norm for v in emb1]
    emb2_norm = pow(sq_norm(emb2), 0.5)
    emb2_normed = [v/emb2_norm for v in emb2]
    # cosine_similarity = 1 - normed_euclidean_distance/2
    return 1 - sq_norm([v1 - v2 for v1, v2 in zip(emb1_normed, emb2_normed)])/2

# === Main =============================================================================================================

class Commands():
    @classmethod
    def initialize_class(cls, BOT_CONFIG_FILE):
        with open(BOT_CONFIG_FILE, 'r') as config_file:
            cls.bot_config = load(config_file)

        for command, value in (cls.bot_config["all_users"] | cls.bot_config["admin_only"]).items():
            setattr(cls, command, value["name"])

    def __init__(self, BOT_CONFIG_FILE, remote_data: RemoteData):
        # kinda ugly but allows a single source of truth shared between python and terraform
        # TODO: dynamically generate docstrings based on description in bot_config

        self.initialize_class(BOT_CONFIG_FILE)

        self.all_users_command_handlers = {
            value["name"]: globals()[value["handler"]](remote_data)
            for command, value in self.bot_config["all_users"].items()
        }

        # take as input a function of update and context, and return that function but with an added check if the sender is the admin
        def add_verification(fn):
            async def add_verification_(update: Update, context: ContextTypes.DEFAULT_TYPE):
                if await ensure_admin(update, context):
                    await fn(update, context)
            return add_verification_

        self.admin_only_command_handlers = {
            value["name"]: add_verification(globals()[value["handler"]](remote_data))
            for command, value in self.bot_config["admin_only"].items()
        }

        self.command_handlers = self.all_users_command_handlers | self.admin_only_command_handlers

def setup_app_polling():
    openai.api_key = getenv("OPENAI_API_KEY")
    BOT_TOKEN = getenv("BOT_TOKEN")
    bucket = getenv("BUCKET")
    BOT_CONFIG_FILE = getenv("BOT_CONFIG_FILE")

    remote_data = RemoteData(bucket)
    
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # add command handlers
    # this initializes Commands as well
    for command, handler_fn in Commands(BOT_CONFIG_FILE, remote_data).command_handlers.items():
        application.add_handler(CommandHandler(command, handler_fn))

    # create message handlers
    regular_message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), 
        regular_message(remote_data))
    document_handler = MessageHandler(filters.ATTACHMENT, handle_document(remote_data))
    unsupported_message_handler = MessageHandler(filters.ALL, unsupported_message)

    # add message handlers
    application.add_handler(regular_message_handler)
    application.add_handler(document_handler)
    application.add_handler(unsupported_message_handler)

    application.add_error_handler(error_handler)

    return application

async def webhook_(request):
    """Webhook entry point. Returns 200 even on failures, because otherwise it will retry in an undesirable way."""
    openai.api_key = getenv("OPENAI_API_KEY")
    BOT_TOKEN = getenv("BOT_TOKEN")
    bucket = getenv("BUCKET")
    BOT_CONFIG_FILE = getenv("BOT_CONFIG_FILE")

    remote_data = RemoteData(bucket)
    bot = ExtBot(BOT_TOKEN)

    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        context = FakeContext(bot, update.message, None)

        # For documents
        if update.message.document:
            handler = handle_document(remote_data)
        elif update.message.text:
            # For commands
            if update.message.text.startswith("/"):
                command = update.message.text.split(" ", 1)[0][1:]
                # this initializes Commands as well
                command_handlers_ = Commands(BOT_CONFIG_FILE, remote_data).command_handlers
                if command in command_handlers_:
                    handler = command_handlers_[command]
                else:
                    handler = invalid_command
            # For regular messages
            else:
                handler = regular_message(remote_data)
        # Nothing else is supported
        else:
            handler = unsupported_message
        try:
            await handler(update, context=context)
            return 'okay', 200
        except FakeTestErrorException as e:
            return Responses.TEST
        except Exception as e:
            context.set_error(e)
            await error_handler(update, context=context)
            # check if context.status is set
            if hasattr(context, "status"):
                return context.status, 200
            else:
                return 'no_status', 200
    else:
        return 'error', 400

def webhook(request):
    return asyncio_run(webhook_(request))

# Set the environment variable LOCAL_MODE if you want to run the bot on a server;
# do not set it if you want to use google cloud functions
if getenv("LOCAL_MODE") is not None:
    if __name__ == "__main__":
        setup_app_polling().run_polling()
