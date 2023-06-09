1. Message @BotFather on Telegram to create a new Telegram bot.

2. [Sign up to get an OpenAI API key.](https://openai.com/blog/openai-api)
This is configured to use GPT-4, which currently requires a waitlist.
If you aren't on the waitlist yet, set the MODEL variable in gpt_4_telegram.py as desired.
It may be wise to [set spending limits](https://platform.openai.com/account/billing/limits) at this time.

3. Create a .env file with the following format:
```
BOT_TOKEN="<your Telegram bot token here>"
OPENAI_API_KEY="<your OpenAI API key here>"
```

4. [Create a GCP account.](https://console.cloud.google.com/welcome).

5. Create a GCP project. In the same directory as `server.tf`, create a file `terraform.tfvars` of the following format:
```hcl
project     = "your-bot-server-129828"
region      = "us-west1"
zone        = "us-west1-a"
user_name   = "your-username-for-ssh"
```

6. Ensure that you have GCP SSH credentials at ~/.ssh/google_compute_engine. (These can be created automatically when you attempt to SSH.)

7. [Install Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) and run `terraform init` and `terraform apply`.
The startup script requires installing some large libraries with `pip`, so it may take several minutes after the `terraform apply` completes for the bot to start.
The bot should automatically restart every 4 hours.
This will cause it to forget previous conversations, but is useful since it is currently very brittle against errors.

8. If you would ever like to clean up your GCP resources, run `terraform destroy`, or just delete the VM instance `bot-server` in the console.
