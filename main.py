import logging
import openai

from asyncio import run as asyncio_run
from datetime import datetime
from google.cloud import storage
from json import dumps, loads
from os import getenv, path, remove
from random import choices as rand_choices
from string import ascii_letters, digits
from telegram import Message, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CallbackContext, ContextTypes, CommandHandler, ExtBot, MessageHandler, filters
from tiktoken import encoding_for_model
from time import time
from typing import Optional


# === Setup ===========================================================================================================

MODEL = "gpt-4"  # "text-davinci-003"
LONG_MODEL = "gpt-4-32k"
FAST_MODEL = "gpt-3.5-turbo"
LONG_FAST_MODEL = "gpt-3.5-turbo-16k"

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
}

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful Telegram chatbot. You can use Telegram markdown message formatting, e.g. `inline code`, ```c++\ncode written in c++```, *bold*, and _italic_."}

TURBO_COMMAND = 'turbo'
NO_RESPONSE_COMMAND = 'no_response'

HELP_MESSAGE = "i'm a chatbot for interfacing with openai's api.\n\nby default, i respond to direct messages with `gpt-4`; you can also use the /turbo command to send a message to `gpt-3.5-turbo` instead (or, if your message is sufficiently long, `gpt-3.5-turbo-16k`).\n\ni can also read documents, which you can send to me as attachments. i'll respond to the document if it has a caption, or just acknowledge receipt if it doesn't.\n\ni remember your messages until you send me a /new\_conversation command, at which point i'll forget everything you've said before."

class Message:
    def __init__(self, message_id, message_text):
        self.message_id = message_id
        self.message_text = message_text

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
    pass

class UnsupportedMessageException(Exception):
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
        
    def read_conversation(self, chat_id):
        source_blob_name = f"{chat_id}/{self.get_live_conversation(chat_id)}"
        blob = self.bucket.blob(source_blob_name)
        logging.info('File {} downloaded'.format(source_blob_name))
        return loads(blob.download_as_bytes())
    
    def append_conversation(self, chat_id, role, content):
        conversation_data = self.read_conversation(chat_id)
        conversation_data.append({"role": role, "content": content})
        self.write_conversation(chat_id, conversation_data)
        logging.info(f"appended to live conversation for chat {chat_id}")

    def start_new_conversation(self, chat_id):
        new_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.set_live_conversation(chat_id, new_id)
        self.write_conversation(chat_id, [SYSTEM_PROMPT])
        logging.info(f"created new conversation for chat {chat_id} with id {new_id}")
        return new_id
    
    def set_verified(self, chat_id, val):
        destination_blob_name = f"{chat_id}/verified"
        blob = self.bucket.blob(destination_blob_name)
        if val:
            # create an empty blob
            blob.upload_from_string("")
        else:
            # delete the blob
            blob.delete()
    
    def is_verified(self, chat_id):
        destination_blob_name = f"{chat_id}/verified"
        blob = self.bucket.blob(destination_blob_name)
        return blob.exists()
    
    def set_live_conversation(self, chat_id, conversation_id: str):
        # check if the chat_id has been verified.
        # if public access is not allowed, raise an exception if it does not (user has not been verified yet)
        if getenv("ALLOW_PUBLIC") is None:
            if not self.is_verified(chat_id):
                logging.info(f"unverified chat {chat_id} tried to send a message")
                raise UnverifiedException(chat_id)

        destination_blob_name = f"{chat_id}/live"
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(conversation_id)
        logging.info(f"set live conversation for chat {chat_id} to {conversation_id}")

    def get_live_conversation(self, chat_id) -> Optional[str]:
        """Returns the id of the conversation that is currently live for the given chat, or None if there is no live conversation."""
        source_blob_name = f"{chat_id}/live"
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
    

# === Telegram stuff ===================================================================================================

# for some reason a real CallbackContext is not working with the webhook setup
class FakeContext(CallbackContext):
    def __init__(self, bot, message, err):
        self._bot = bot
        self._error = err

        # if message starts with '/', it's a command
        if message.text.startswith("/"):
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.debug("adding new chat")
    await message_user(update.effective_chat.id, "welcome! please wait to be verified. (a human has to do it, so it might be slow)", context)

    # message the admin
    await admin_message(f"new chat!\nuser info: {update.effective_chat}\nchat id:", context)
    await admin_message(f"{update.effective_chat.id}", context)

def verify(remote_data: RemoteData):
    # verify the chat with the given id
    async def verify_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # check if the sender is the admin
        if update.effective_chat.id != int(getenv("ADMIN_CHAT_ID")):
            logging.info(f"unauthorized user {update.effective_chat.id} tried to verify a chat")
            await message_user(update.effective_chat.id, "nice try", context)
            return
        
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
    
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.debug("sinding help message")
    await message_user(update.effective_chat.id, HELP_MESSAGE, context)

def new_conversation(remote_data: RemoteData):
    async def new_conversation_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        remote_data.start_new_conversation(update.effective_chat.id)
        logging.debug("new conversation started")
        await message_user(update.effective_chat.id, "new conversation started", context)
    return new_conversation_

async def record_user_message(remote_data: RemoteData, update: Update, process_msg=lambda msg: msg.text):
    current_chat_id = update.effective_chat.id
    live_conversation_id = remote_data.get_live_conversation(current_chat_id)
    if live_conversation_id is None:
            live_conversation_id = remote_data.start_new_conversation(current_chat_id)
    remote_data.append_conversation(current_chat_id, "user", process_msg(update.message))
    logging.info(f"recorded message from chat {current_chat_id} with id {live_conversation_id}")

async def respond_conversation(remote_data: RemoteData, current_chat_id: int, context: ContextTypes.DEFAULT_TYPE, model=MODEL):
    chat_history = remote_data.read_conversation(current_chat_id)

    # use the long context model if necessary; error if that's not sufficient
    max_tokens = MODEL_DATA[model]["max_tokens"]
    enc = encoding_for_model(model)
    # this isn't the exact number of tokens, but it's kind of unclear how to actually get that value.
    # this should at least err in the direction of overcounting tokens, which is ok since we need some space for a response.
    token_num = len(enc.encode(dumps(chat_history)))
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
    return text.startswith(f"/{TURBO_COMMAND}")

def remove_turbo(text: str):
    return text[len(TURBO_COMMAND) + 2:]

def turbo_message(remote_data: RemoteData):
    async def turbo_message_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info("turbo message received")
        await regular_message_core(remote_data, update, context, process_msg=lambda msg: remove_turbo(msg.text), model=FAST_MODEL)
    return turbo_message_

def no_response(remote_data: RemoteData):
    async def no_response_(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.info(f"message with no response requested received at {time()}: {update}")
        def removenoresponsecommand(msg):
            return msg.text[len(NO_RESPONSE_COMMAND) + 2:]

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
            await message_user(current_chat_id, "document received", context)
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

async def unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await message_user(update.effective_chat.id, "sorry, i don't know how to handle that type of message", context)

async def unsupported_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"unsupported message received at {time()}: {update}")
    await unsupported_message(update, context)

async def error_handler(update: Optional[object], context: CallbackContext):
    """Log the error and send a telegram message to notify the user."""
    logging.error(msg="Exception while handling an update:", exc_info=context.error)
    if update is not None:
        # Check if it was an UnverifiedException
        if isinstance(context.error, UnverifiedException):
            logging.info(f"unverified chat {update.effective_chat.id} tried to send a message")
            await message_user(update.effective_chat.id, "please send /start to get verified", context)

        elif isinstance(context.error, UnsupportedMessageException):
            logging.info("message is not a document")
            await message_user(update.effective_chat.id, "sorry, i can only read documents", context)

        else:
            await message_user(update.effective_chat.id, f"An error occurred: {context.error}", context)
    else:
        logging.error("update is None")


# === OpenAI stuff =====================================================================================================

def query_model(previous_messages, model=MODEL):
    response = openai.ChatCompletion.create(model=model, messages=previous_messages)
    logging.info(f"got response: {response}")
    return response['choices'][0]['message']['content']


# === Main =============================================================================================================

def command_handlers(remote_data: RemoteData) -> dict[str, CommandHandler]:
    return {
    "start": start,
    "help": help,
    "new_conversation": new_conversation(remote_data),
    TURBO_COMMAND: turbo_message(remote_data),
    NO_RESPONSE_COMMAND: no_response(remote_data),
    "verify": verify(remote_data),
}

def setup_app_polling():
    openai.api_key = getenv("OPENAI_API_KEY")
    BOT_TOKEN = getenv("BOT_TOKEN")
    bucket = getenv("BUCKET")

    remote_data = RemoteData(bucket)
    
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # add command handlers
    for command, handler_fn in command_handlers(remote_data).items():
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

def webhook(request):
    openai.api_key = getenv("OPENAI_API_KEY")
    BOT_TOKEN = getenv("BOT_TOKEN")
    bucket = getenv("BUCKET")

    remote_data = RemoteData(bucket)
    bot = ExtBot(BOT_TOKEN)

    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        context = FakeContext(bot, update.message, None)

        # For documents
        if update.message.document:
            handler = handle_document
        # For commands
        elif update.message.text.startswith("/"):
            command = update.message.text.split(" ", 1)[0][1:]
            command_handlers_ = command_handlers(remote_data)
            if command in command_handlers_:
                handler = command_handlers_[command]
        # For regular messages
        elif update.message.text:
            handler = regular_message(remote_data)
        # Nothing else is supported
        else:
            handler = unsupported_message_handler
        try:
            asyncio_run(handler(update, context=context))
            return 'okay', 200
        except Exception as e:
            context.set_error(e)
            asyncio_run(error_handler(update, context=context))
            return str(e), 500
    else:
        return 'error', 400


# Set the environment variable LOCAL_MODE if you want to run the bot on a server;
# do not set it if you want to use google cloud functions
if getenv("LOCAL_MODE") is not None:
    if __name__ == "__main__":
        setup_app_polling().run_polling()
