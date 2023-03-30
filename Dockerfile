# See https://stackoverflow.com/questions/68673221/warning-running-pip-as-the-root-user
# for enhancements to Dockerfile e.g. not running as root & in venv

FROM python:3.10-slim-bullseye
RUN apt-get update && apt-get install -y

EXPOSE 8080

RUN pip install -U pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy into a directory of its own (so it isn't in the toplevel dir)

# Copy required directories
COPY docs app/docs
COPY src app/src
COPY settings.toml app/
COPY .streamlit/config.toml app/.streamlit/config.toml


WORKDIR /app 

# Run it!

ENTRYPOINT ["streamlit", "run", "src/Main.py", "--server.port=8080", "--server.address=0.0.0.0"]
