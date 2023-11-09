import toml
from app.server_fastapi import app
import uvicorn
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="0.0.0.0")
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    uvicorn.run(
        app, host=args.host, port=args.port)
