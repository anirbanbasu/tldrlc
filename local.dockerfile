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
FROM python:slim-bookworm

# Upgrade and install basic packages
RUN apt-get update && apt-get -y upgrade && apt-get -y install nano build-essential && apt-get -y autoremove

# Create a non-root user
RUN useradd -m -u 1000 app_user

ENV HOME="/home/app_user"

USER app_user
# Set the working directory in the container
WORKDIR $HOME/app

# Copy only the requirements file to take advantage of layering (see: https://docs.cloud.ploomber.io/en/latest/user-guide/dockerfile.html)
COPY ./requirements.txt ./requirements.txt

# Setup Virtual environment
ENV VIRTUAL_ENV="$HOME/app/venv"
RUN python -m venv $VIRTUAL_ENV
RUN $VIRTUAL_ENV/bin/python -m ensurepip
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip setuptools

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY ./*.md ./LICENSE ./*.py ./*.sh ./*.css ./
COPY ./.env.docker /.env
COPY ./ui/*.py ./ui/
COPY ./utils/*.py ./utils/

# Expose the port to conect
EXPOSE 8765
# Run the application
ENTRYPOINT [ "/home/app_user/app/run_starlette.sh" ]