terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }
}


provider "google" {
    project     = "bot-server-383318"
    region      = "us-west1"
    zone        = "us-west1-a"
}

locals {
    script_directory = "/home/evan"
    script_name = "bot_script.py"
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
      pip3 install dotenv

      # Set up the cron job to execute the Python script every 4 hours
      (crontab -l 2>/dev/null; echo "0 */4 * * * cd ${local.script_directory} && python3 ${local.script_name}") | crontab -
  EOF

    provisioner "file" {
        source = "gpt_4_telegram.py"
        destination = "${local.script_directory}/${local.script_name}"
        connection {
            type = "ssh"
            user = "evan"
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
