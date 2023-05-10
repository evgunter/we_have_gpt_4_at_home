import os
import dotenv
import logging
from telegram.constants import ParseMode
import telegram.ext
import time
import openai


# === Setup ============================================================================================================

MODEL = "gpt-4"  # "gpt-3.5-turbo"  # "text-davinci-003"
LONG_MODEL = "gpt-4-32k"
FAST_MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful Telegram chatbot. You can use Telegram markdown message formatting, e.g. `inline code`, ```c++\ncode written in c++```, *bold*, and _italic_."}

TURBO_COMMAND = 'turbo'

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
        logging.FileHandler(f'gpt_4_telegram_{time.time()}.log'),
    ]
)

conversations = {}  # current conversations for each chat

# === Telegram stuff ===================================================================================================

async def start(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.debug("adding new chat")
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="welcome!",
                                   parse_mode=ParseMode.MARKDOWN,
     )  # reply_to_message_id=update.message.message_id)

def reset_conversation(chat_id):
    conversations[chat_id] = [SYSTEM_PROMPT]  # reset conversation

async def new_conversation(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    reset_conversation(update.effective_chat.id)
    logging.debug("new conversation started")
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="new conversation started",
                                   parse_mode=ParseMode.MARKDOWN,
    )  # reply_to_message_id=update.message.message_id)

async def regular_message(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE, process_msg=lambda msg: msg, model=MODEL):
    logging.info(f"update received at {time.time()}: {update}")
    
    current_chat_id = update.effective_chat.id

    if current_chat_id not in conversations:
        reset_conversation(update.effective_chat.id)
        logging.info("chat missing!")
        await context.bot.send_message(chat_id=current_chat_id,
                                       text="looks like i've lost the previous conversation! i've started a new one. send your message again, including any context from the previous conversation.",
                                       parse_mode=ParseMode.MARKDOWN,
        )  # reply_to_message_id=update.message.message_id)
        return
    
    conversations[current_chat_id].append({"role": "user", "content": process_msg(update.message.text)})
    logging.info(f"conversations: {conversations}")
    try:
        response = query_model(conversations[current_chat_id], model=model)
    except Exception as e:
        # TODO: if the model fails for length, use the long context model.
        #   also, should maybe give some option to remove the last response so it doesn't ruin the whole conversaion,
        #   if that turns out to be a problem
        logging.error(e)
        await context.bot.send_message(chat_id=current_chat_id,
                                       text=f"model failed with error: {e}",
                                       # parse mode is plain text because a common error is that the model returns an invalid message
        )  # reply_to_message_id=update.message.message_id)
        return
    conversations[current_chat_id].append({"role": "assistant", "content": response})

    await context.bot.send_message(chat_id=current_chat_id,
                                   text=response,
                                   parse_mode=ParseMode.MARKDOWN,
    )  # reply_to_message_id=update.message.message_id)

async def turbo_message(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.info("turbo message received")
    def removeturbo(msg):
        return msg[len(TURBO_COMMAND) + 2:]

    await regular_message(update, context, process_msg=removeturbo, model=FAST_MODEL)


# === OpenAI stuff =====================================================================================================

def query_model(previous_messages, model=MODEL):
    response = openai.ChatCompletion.create(model=model, messages=previous_messages)
    logging.info(f"got response: {response}")
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    dotenv.load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    BOT_TOKEN = os.getenv("BOT_TOKEN")

    application = telegram.ext.ApplicationBuilder().token(BOT_TOKEN).build()

    start_handler = telegram.ext.CommandHandler('start', start)
    new_conversation_handler = telegram.ext.CommandHandler('new_conversation', new_conversation)
    turbo_handler = telegram.ext.CommandHandler(TURBO_COMMAND, turbo_message)

    regular_message_handler = telegram.ext.MessageHandler(telegram.ext.filters.TEXT & (~telegram.ext.filters.COMMAND),
                                                          regular_message)

    application.add_handler(start_handler)
    application.add_handler(regular_message_handler)
    application.add_handler(new_conversation_handler)
    application.add_handler(turbo_handler)

    application.run_polling()
