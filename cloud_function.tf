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
  sensitive = true
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
  sensitive = true
}


# === TELEGRAM RESOURCES ===

resource "telegram_bot_webhook" "gptathome_webhook" {
  url = google_cloudfunctions_function.function.https_trigger_url
  # update types are at https://core.telegram.org/bots/api#update
  allowed_updates = ["message"]
}

data "local_file" "commands" {
  filename = local.bot_config_file
}

resource "telegram_bot_commands" "gptathome_commands" {
  # load the commands from the file commands.json, taking the "name" field as the key and "description" field as the value
  commands = concat(local.all_users_desc, local.models_desc)
}


# === SETUP RESOURCES ===

locals {
    bot_config_file = "commands.json"

    code_files  = [
        "main.py",
        "requirements.txt",
        local.bot_config_file,
    ]
    source_code_sha1 = sha1(join("", [for f in local.code_files : filesha1(f)]))

    zipped_code_local = "source_local.zip"
    zipped_code_remote = "source_remote.zip"

    config = jsondecode(data.local_file.commands.content)
    models_desc = [for _, v in local.config["models"] : { command = v["name"], description = "send a message to ${v["model"]} instead of ${local.config["default"]["model"]}" }]
    all_users_desc = [for _, v in local.config["all_users"] : { command = v["name"], description = v["description"] }]
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
    BOT_CONFIG_FILE = local.bot_config_file
  }
}

resource "google_cloudfunctions_function_iam_member" "invoker" {
  project        = google_cloudfunctions_function.function.project
  region         = google_cloudfunctions_function.function.region
  cloud_function = google_cloudfunctions_function.function.name

  role   = "roles/cloudfunctions.invoker"
  member = "allUsers"
}


# === OUTPUTS ===

output "bucket_name" {
  value = google_storage_bucket.bucket.name
}

output "webhook_url" {
  value = google_cloudfunctions_function.function.https_trigger_url
}

output "environment_variables" {
  value = join("", [for k, v in google_cloudfunctions_function.function.environment_variables : "export ${k}=${v}\n"])
  description = "environment variables formatted as export VAR=value"
  sensitive = true
}
