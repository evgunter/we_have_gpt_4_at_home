To use, create a .env file with the following format:
```
BOT_TOKEN="<your Telegram bot token here>"
OPENAI_API_KEY="<your OpenAI API key here>"
```
Then install Terraform and run `terraform init` and `terraform apply`.
The script should automatically restart every 4 hours, but you can also 