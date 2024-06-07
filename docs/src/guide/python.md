# Python dependencies

You will need Python installed on your computer. The code in this repository has been tested on Python 3.12.0. Refrain from using the system Python. Use a Python version and virtual environment manager such as [pyenv](https://github.com/pyenv/pyenv). Create a new Python virtual environment for Python 3.12.0 or above. You can install all the dependencies for this application in your virtual environment by running the following.

```
pip install --upgrade pip
pip install -r requirements.txt
```
## Optional: Uninstall all dependencies to start afresh

If necessary, you can uninstall everything previously installed by `pip` (in a virtual environment) by running the following.

```
pip freeze | xargs pip uninstall -y
```

## Upgrading Python dependencies

The currently installed packages can be upgraded and the `requirements.txt` updated accordingly by running the following.

```
sed 's/==.*$//' requirements.txt | xargs pip install --upgrade
pip-autoremove -f > requirements.txt
```
