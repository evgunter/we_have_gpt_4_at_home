terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }
}

variable "project" {
  description = "The project ID"
  type = string
}

variable "region" {
  description = "The region to deploy to"
  type = string
}

variable "zone" {
  description = "The zone to deploy to"
  type = string
}

variable "user_name" {
    description = "The username to SSH in with"
    type = string
}

provider "google" {
    project     = var.project
    region      = var.region
    zone        = var.zone
}

locals {
    script_directory = "/home/${var.user_name}"
    bot_script_name = "bot_script.py"
    restart_script_name = "restart_bot.sh"
}

# SSH in with `gcloud compute ssh bot-server --zone us-west1-a`
resource "google_compute_instance" "default" {
    name = "bot-server"
    machine_type = "f1-micro"
    zone = "us-west1-a"

    # Debug with `google_metadata_script_runner startup`
    metadata_startup_script = <<-EOF
      #!/bin/bash

      # Move to directory containing script created by the file provisioner
      cd ${local.script_directory}

      # Install required packages for Python script
      sudo apt update
      sudo apt -y install python3 python3-pip
      pip3 install python-dotenv python-telegram-bot openai

      chmod +x ${local.restart_script_name}

      # Set up the cron job to restart the script at 4am every day
      (crontab -l 2>/dev/null; echo "0 4 * * * cd ${local.script_directory} && ./${local.restart_script_name}") | crontab -

      # Start the Python script for the first time
      python3 ${local.bot_script_name}
  EOF

    provisioner "file" {
        source = local.bot_script_name
        destination = "${local.script_directory}/${local.bot_script_name}"
        connection {
            type = "ssh"
            user = var.user_name
            private_key = file("~/.ssh/google_compute_engine")
            host = self.network_interface[0].access_config[0].nat_ip
        }
    }
    provisioner "file" {
        source = ".env"
        destination = "${local.script_directory}/.env"
        connection {
            type = "ssh"
            user = var.user_name
            private_key = file("~/.ssh/google_compute_engine")
            host = self.network_interface[0].access_config[0].nat_ip
        }
    }
    provisioner "file" {
        source = local.restart_script_name
        destination = "${local.script_directory}/${local.restart_script_name}"
        connection {
            type = "ssh"
            user = var.user_name
            private_key = file("~/.ssh/google_compute_engine")
            host = self.network_interface[0].access_config[0].nat_ip
        }
    }

    boot_disk {
        initialize_params {
            image = "debian-cloud/debian-11"
        }
    }
    network_interface {
        network = "default"
        access_config {}
    }
}
