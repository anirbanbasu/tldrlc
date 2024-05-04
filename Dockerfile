# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pull Python 3.12 on Debian Bookworm slim image
FROM python:3.12.3-slim-bookworm

# Upgrade and install basic packages
RUN apt-get update && apt-get -y upgrade && apt-get -y dist-upgrade && apt-get -y install nano build-essential

# Set the working directory in the container
WORKDIR /app

# Copy the project files
COPY ./*.py ./
COPY ./pages/*.py ./pages/
COPY ./utils/*.py ./utils/
COPY ./*.md ./LICENSE ./requirements.txt ./
COPY ./.env.docker ./.env
COPY ./*.sh ./
RUN chmod +x *.sh

# Setup Virtual environment
RUN python -m venv /app/venv
RUN /app/venv/bin/python -m ensurepip
RUN /app/venv/bin/pip install --no-cache --upgrade pip setuptools

ENV PATH="/app/venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/venv"

# Install dependencies
RUN /app/venv/bin/pip install -r requirements.txt

EXPOSE 8765
# Run the application
# CMD ["solara", "run", "solapp.py", "--host=0.0.0.0", "--production"]
ENTRYPOINT [ "/app/run_starlette.sh" ]
# Expose the port to conect