# === VARIABLES AND PROVIDERS ===

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
    telegram = {
      source = "yi-jiayu/telegram"
      version = ">= 0.3.1"
    }
    random = {
      source = "hashicorp/random"
      version = ">= 3.1.0"
    }
  }
}

# --- telegram ---

variable "telegram_bot_token" {
  description = "The bot token from BotFather"
  type = string
}

provider "telegram" {
  bot_token = var.telegram_bot_token
}


variable "admin_id" {
    description = "The Telegram user ID of the bot admin"
    type = string
}

variable "allow_public" {
    description = "Whether to allow anyone to use the bot without requiring admin approval"
    type = bool
}

# --- google ---

variable "project" {
  description = "The project ID"
  type        = string
}

variable "region" {
  description = "The region to deploy to"
  type        = string
}

variable "zone" {
  description = "The zone to deploy to"
  type        = string
}

provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}


# --- openai ---

variable "openai_api_key" {
  description = "OpenAI API key"
  type = string
}


# === TELEGRAM RESOURCES ===

resource "telegram_bot_webhook" "gptathome_webhook" {
  url = google_cloudfunctions_function.function.https_trigger_url
  # update types are at https://core.telegram.org/bots/api#update
  allowed_updates = ["message"]
}

resource "telegram_bot_commands" "gptathome_commands" {
  commands = [
    {
      command = "start",
      description = "start the bot"
    },
    {
      command = "help",
      description = "send usage message"
    },
    {
      command = "new_conversation",
      description = "start a new conversation, forgetting the previous one"
    },
    {
      command = "turbo",
      description = "send a message to gpt-3.5-turbo instead of gpt-4"
    },
    {
      command = "no_response",
      description = "send a message without requesting a response (e.g. to break up one long message into parts)"
    },
    {
      command = "switch_conversation",
      description = "send the date and time you sent the /new_conversation command for a previous conversation in YYYY-mm-DD-HH-MM format to switch to a different conversation"
    }
  ]
}

# === SETUP RESOURCES ===

locals {
    code_files  = [
        "main.py",
        "requirements.txt"
    ]
    source_code_sha1 = sha1(join("", [for f in local.code_files : filesha1(f)]))

    zipped_code_local = "source_local.zip"
    zipped_code_remote = "source_remote.zip"
}

# zip the code for the cloud function
resource "null_resource" "zip" {
  triggers = {
    # reupload the source code if it changes
    source_code_sha1 = local.source_code_sha1
  }

  provisioner "local-exec" {
    command = "zip -r ${local.zipped_code_local} ${join(" ", local.code_files)}"
  }
}

resource "random_id" "bucket_id" {
  byte_length = 8
}


# === GOOGLE RESOURCES ===

resource "google_storage_bucket" "bucket" {
  name     = "gptathome-bucket-${random_id.bucket_id.hex}"
  location = var.region
}

resource "google_storage_bucket_object" "archive" {
  # include the sha1 hash in the name so that the function is recreated when the source code changes
  name   = "${local.source_code_sha1}-${local.zipped_code_remote}"
  bucket = google_storage_bucket.bucket.name
  source = local.zipped_code_local
}

resource "google_cloudfunctions_function" "function" {
  name                  = "gptathome"
  description           = "runs gpt-4 telegram bot"
  runtime               = "python39"
  available_memory_mb   = 1024
  source_archive_bucket = google_storage_bucket.bucket.name
  source_archive_object = google_storage_bucket_object.archive.name
  trigger_http          = true
  timeout               = 5 * 60
  entry_point           = "webhook"

  environment_variables = {
    OPENAI_API_KEY = var.openai_api_key
    BOT_TOKEN = var.telegram_bot_token
    BUCKET = google_storage_bucket.bucket.name
    ADMIN_CHAT_ID = var.admin_id
    ALLOW_PUBLIC = var.allow_public
  }
}

resource "google_cloudfunctions_function_iam_member" "invoker" {
  project        = google_cloudfunctions_function.function.project
  region         = google_cloudfunctions_function.function.region
  cloud_function = google_cloudfunctions_function.function.name

  role   = "roles/cloudfunctions.invoker"
  member = "allUsers"
}
