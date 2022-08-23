import toml
from pathlib import Path

def replace_string_in_file(str_in, str_replace, file_text):
    return [line.replace(str(str_in), str(str_replace)) for line in file_text]


def create_dockerfile_with_env_vars(python_version="3.10", env_file=".env_dockerfile.toml", dockerfile="Dockerfile", sections=None):
    try:
        env_dict = toml.load(Path(env_file))
    except Exception as IOError:
        raise(IOError)

    if sections is None:
        sections = DOCKERFILE_SECTIONS = [
            "head", "env", "port", "py", "app"]

    # Read in each of the Dockerfile pieces

    if Path(dockerfile).exists():
        Path(dockerfile).unlink()

    n_secret = len("SECRET_")

    for section in sections:
        try:
            env_dict = toml.load(Path(env_file))
        except Exception as IOError:
            raise(IOError)
        if section != "env":
            try:
                with open(Path(f"Dockerfile_{section}.txt"), "r") as f:
                    text = f.readlines()
                    with open(Path(dockerfile), "a") as fout:
                        for k, v in env_dict.items():
                            if k[:n_secret] != "SECRET_":
                                k = f"{{{{{k}}}}}"   # Looking for {{TEXT}} to replace
                                text = replace_string_in_file(k, v, text)
                        fout.writelines(text)
                        fout.write("\n")
            except Exception as IOError:
                raise(IOError)
        else:
            with open(Path(dockerfile), "a") as fout:
                for k, v in env_dict.items():
                    if k[:n_secret] == "SECRET_":
                        k = k.replace("SECRET_", "")
                        fout.writelines(f"ENV {k}={v}\n")
                fout.write("\n")


if __name__ == "__main__":
    create_dockerfile_with_env_vars(python_version="3.10")
