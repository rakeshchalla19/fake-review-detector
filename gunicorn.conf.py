"""Also applies when an existing Render service starts with `gunicorn app:app`."""
import os
bind = '0.0.0.0:' + os.getenv('PORT', '5000')
workers = 1
threads = 2
timeout = 120
accesslog = '/dev/null'
