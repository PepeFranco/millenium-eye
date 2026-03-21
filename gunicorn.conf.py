"""
Gunicorn configuration for production.

  --preload   loads the app (and the ORB index) in the master process before
              forking, so workers share the index memory instead of each
              loading their own copy.
"""

import multiprocessing

# Load app + ORB index once before forking workers
preload_app = True

# 2 processes × 8 threads = 16 concurrent requests
# gthread worker releases the GIL during OpenCV C++ computation
workers     = 2
threads     = 8
worker_class = "gthread"

# Unix socket — Nginx proxies to this
bind = "unix:/tmp/millenium-eye.sock"

# Generous timeout: ORB matching can take ~700ms, allow headroom
timeout = 60

# Logging
accesslog = "-"
errorlog  = "-"
loglevel  = "info"

# Increase max request body size to 5 MB (for 1280px JPEG frames)
limit_request_line  = 0
limit_request_field_size = 0
