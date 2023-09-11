# Setup

1. Message @BotFather on Telegram to create a new Telegram bot. You will need the bot token later.

2. [Sign up to get an OpenAI API key.](https://openai.com/blog/openai-api)
It may be wise to [set spending limits](https://platform.openai.com/account/billing/limits) at this time.

3. [Create a GCP account.](https://console.cloud.google.com/welcome)

4. Create a GCP project.
[Enable the Cloud Functions API](https://console.developers.google.com/apis/api/cloudfunctions.googleapis.com/overview) and [Cloud Build API](https://console.developers.google.com/apis/api/cloudbuild.googleapis.com/overview) for your new project.

5. Message `/start` to @userinfobot on Telegram to get your Telegram user ID.

6. In the main directory, create a file `terraform.tfvars` with the following format:
```hcl
project            = "your-project"
region             = "us-west1"
zone               = "us-west1-a"
user_email         = "your-google-email@example.com"

telegram_bot_token = "your:bot-token"
openai_api_key     = "your-api-key"

admin_id           = "your-telegram-user-id"
allow_public       = false
```
You may set allow_public to true if you would like anyone to be able to message the bot without approval.

7. [Install Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) and run `terraform init` and `terraform apply`.
When the command finishes running, your bot should be ready to message!

8. If you would ever like to clean up the created resources, run `terraform destroy`.

# Testing

1. Uncomment the `pytest` line in `requirements.txt`.
(You should re-comment it when deploying, since it's unnecessary then.)

2. Run `pip install -r requirements.txt`.

3. If you would like to run the webhook test suite, run `pytest -s test.py`.
If you would like to run locally with polling, just run the file with `python main.py`.

