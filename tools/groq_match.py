#!/usr/bin/env python3
"""Send Nyaa titles to Groq for anime title matching using OpenAI SDK."""

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    os.system("pip install openai -q")
    from openai import OpenAI

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    with open(".env") as f:
        for line in f:
            if line.startswith("GROQ_API_KEY="):
                GROQ_API_KEY = line.split("=", 1)[1].strip()
                break

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

SYSTEM_PROMPT = open("data/training/nyaa_titles_system_prompt.md").read()
USER_PROMPT = open("data/training/nyaa_titles_input.txt").read()

print("Sending to Groq...")
response = client.responses.create(
    model="llama-3.3-70b-versatile",
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    temperature=0.1,
)

content = response.output_text
print("Response received, saving...")
with open("data/training/nyaa_titles_matched.txt", "w") as f:
    f.write(content)

print(f"Saved to data/training/nyaa_titles_matched.txt")
print(f"Response length: {len(content)} chars")
