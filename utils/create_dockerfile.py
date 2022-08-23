import toml
from pathlib import Path


def replace_string_in_file(str_in, str_replace, file_text):
    text = [line.replace(str(str_in), str(str_replace)) for line in file_text]
    replaced = text != file_text
    return replaced, text


def create_dockerfile_with_env_vars(python_version="3.10", env_file=".env_dockerfile.toml", dockerfile="Dockerfile", secrets_file=".secrets.toml", sections=None):
    try:
        env_dict = toml.load(Path(env_file))
    except Exception as IOError:
        raise (IOError)

    try:
        secrets_dict = toml.load(Path(secrets_file))
    except Exception as IOError:
        raise (IOError)

    if sections is None:
        sections = DOCKERFILE_SECTIONS = ["head", "env", "port", "py", "app"]

    # Delete any existing dockerfile

    if Path(dockerfile).exists():
        Path(dockerfile).unlink()

    # Read in each of the Dockerfile pieces

    for section in sections:
        if section != "env":
            try:
                with open(Path(f"Dockerfile_{section}.txt"), "r") as f:
                    text = f.readlines()
                    with open(Path(dockerfile), "a") as fout:
                        for k, v in env_dict.items():
                            # Looking for {{TEXT}} to replace
                            k = f"{{{{{k}}}}}"
                            replaced, text = replace_string_in_file(k, v, text)
                            if replaced:
                                print(f"\nReplaced: {k} with {v} in section {section}")
                        fout.writelines(text)
                        fout.write("\n")
            except Exception as IOError:
                raise (IOError)
        else:  # section = "env"
            with open(Path(dockerfile), "a") as fout:
                for k, v in secrets_dict.items():
                    # all secrets from .secrets.toml are written as ENV variables
                    print(f"\nSecret: ENV {k}={v}")
                    fout.writelines(f"ENV {k}={v}\n")
                fout.write("\n")


if __name__ == "__main__":
    create_dockerfile_with_env_vars()
