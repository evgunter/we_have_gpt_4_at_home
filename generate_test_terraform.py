from os import mkdir, path

def convert_file(original_file_path, new_file_path):
    with open(original_file_path, "r") as original_file:
        lines = original_file.readlines()

    with open(new_file_path, "w") as new_file:
        # make sure every edit is made exactly once
        edits_made = {
            "code_file_path_sha1": False,
            "commands_file_path": False,
            "zipped_code_local": False,
            "unstable": False,
        }
        for line in lines:
            if 'source_code_sha1 = sha1(join("", [for f in local.code_files : filesha1(f)]))' in line:
                if edits_made["code_file_path_sha1"]:
                    raise Exception("code_file_path_sha1 was already edited")
                edits_made["code_file_path_sha1"] = True
                new_file.write(line.replace("filesha1(f)", 'filesha1("../${f}")'))
            elif 'bot_config_file = "${path.module}/commands.json' in line:
                if edits_made["commands_file_path"]:
                    raise Exception("commands_file_path was already edited")
                edits_made["commands_file_path"] = True
                new_file.write(line.replace("commands.json", "../commands.json"))
            elif "command = \"zip -r ${local.zipped_code_local} ${join(\" \", local.code_files)}\"" in line:
                if edits_made["zipped_code_local"]:
                    raise Exception("zipped_code_local was already edited")
                edits_made["zipped_code_local"] = True
                new_file.write(line.replace("\"zip -r ${local.zipped_code_local} ${join(\" \", local.code_files)}\"", "\"${join(\"\", [for f in local.code_files : \"cp ../${f} . &&\"])} zip -r ${local.zipped_code_local} ${join(\" \", local.code_files)}\""))
            elif "name                  = \"gptathome\"" in line:
                if edits_made["unstable"]:
                    raise Exception("unstable was already edited")
                edits_made["unstable"] = True
                new_file.write(line.replace("\"gptathome\"", "\"gptathome-unstable\""))
            else:
                new_file.write(line)
        for key in edits_made:
            if not edits_made[key]:
                raise Exception(f"{key} was not edited")
        

# create .test_env if it doesn't already exist
if not path.exists(".test_env"):
    mkdir(".test_env")

convert_file("cloud_function.tf", ".test_env/cloud_function_unstable.tf")
