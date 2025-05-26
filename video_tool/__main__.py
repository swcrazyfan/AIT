from dotenv import load_dotenv
load_dotenv() # Load .env file at the very beginning

from video_tool.cli import app

if __name__ == "__main__":
    app()