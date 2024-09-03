from flask import Flask
from config import config_by_name
import os

app = Flask(__name__)

config_name = os.getenv('FLASK_CONFIG', 'development')
app.config.from_object(config_by_name[config_name])

from app.routes import *  # Import routes
