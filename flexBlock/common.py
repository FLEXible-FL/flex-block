import os

DEBUG = os.getenv("DEBUG")
DEBUG = int(DEBUG) if DEBUG else 0