"""
Vercel Serverless Entrypoint for El GPT FastAPI Application.
"""

import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server.app import app

# Vercel supports both ASGI 'app' and 'handler'
handler = app
