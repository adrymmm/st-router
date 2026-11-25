import textwrap
import time

from st_router.paths import get_rootdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from dotenv import load_dotenv
import groq
from groq import Groq
from time import sleep
import json

def make_prompt() -> str:
    """Creates the system prompt used by smart-terminal"""

    role = "You are a shell command predictor embedded in the user's terminal."

    objective = "The user typed something into their prompt or is empty ; your job is to produce the command they most likely want to run next."
    workflow = textwrap.dedent("""\
    RECENT INTERACTIONS
    When prior interactions in this folder are included, treat them as the user's working session. Use them to:
    - Resolve references like 'undo that', or 'now do it on the other branch'.

    LEARNING FROM ACCEPTANCE
    You have two sources of truth about this user:
    - Recent interactions: commands you previously suggested in this project.
    - Shell history: commands the user actually executed in their terminal.

    Cross-reference them. For each prior suggestion, find what happened next in the shell history:
    - Ran verbatim → the suggestion landed. Keep doing what worked: same tool, same flags, same shape.
    - Ran with edits → the suggestion was close but wrong on specifics. The edits are the correction. If they added `-i`, they want interactivity; if they swapped `grep` for `rg`, that's their tool; if they changed the target, your scoping was off. Carry the edit forward, not the original.
    - Not run, something else ran instead → the suggestion was rejected. Whatever they ran instead is what they actually wanted for that intent. Treat your suggestion as a negative example.

    INPUT MODES
    The user's input arrives in one of three forms — figure out which:

    1. PARTIAL COMMAND — they started typing a shell command and stopped.Complete it.
    2. NATURAL LANGUAGE — they typed a description in plain English (or any language.Translate it into the command they meant.
    3. EMPTY BUFFER - they havnet typed anything predict the next command based on history and recent interactions

    Never suggest the same command as the last suggestion in this project's interaction history.
    """)

    tooluse= "Before answering, evaluate the user's intent against the descriptions and guidance for the tools actually available to you. Call a matching tool before generating the command. If no available tool applies, answer without a tool call."
    context = "Use the supplied shell, OS, current directory, and recent history to choose an appropriate command."
    completion = "Reply with only the command, on one line, with no explanation."

    headings = ["ROLE", "OBJECTIVE", "WORKFLOW", "TOOL-USE RULES", "ENVIRONMENT CONTEXT", "COMPLETION POLICY"]
    prompt_parts = [role, objective, workflow, tooluse, context, completion]
    prompt = "Agent role: command predictor\n\n" + "\n\n".join(
        f"[{h}]\n\n{p}" for h, p in zip(headings, prompt_parts)
        )
    return prompt


def create_random_message(row: pd.Series) -> str:
    """Expects a row, returns a message as a string"""
    cmd_list = []
    for i in range(10, 0, -1):
        curr = row[f"prev_{i}"]
        if pd.notna(curr):
            cmd_list.append(curr)
    return "Context:\n" + json.dumps({"cwd": row["wd"], "history": cmd_list}, indent=2)

def run_llm(df: pd.DataFrame, prompt: str, model: str = "openai/gpt-oss-20b",
             reasoning_effort: str | None = "low", temperature: float = 0, sleep_s: float = 2) -> list[dict]:
    """Runs llm, on the dataframe containing context window"""
    client = Groq(max_retries=0)

    preds = []
    for i, (idx, row) in enumerate(df.iterrows()): 
        sleep(sleep_s)
        msg = create_random_message(row)
        try:
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": prompt,
                            },
                            {
                                "role": "system",
                                "content": msg,
                            },
                            {
                                "role": "user",
                                "content": "User input:\n"

                            }

                        ],
                        model=model,
                        reasoning_effort=reasoning_effort,
                        temperature=temperature,
                    )
                    latency_s = time.perf_counter() - t0
                    break
                except groq.RateLimitError:
                    if attempt == 4:
                        raise
                    print(f"row {i}: rate limited, waiting {15 * (attempt + 1)}s")
                    sleep(15 * (attempt + 1))
            row_dict = {"idx" : idx,
                        "actual" : row["cmd"],
                        "prediction" : chat_completion.choices[0].message.content,
                        "message" : msg,
                        "raw" : chat_completion.to_dict(),
                        "latency_s" : latency_s 
                        }
            preds.append(row_dict)
            if (i % 10 == 0): print(f"row {i} processed")
        except Exception as e:
            row_dict = {"idx" : idx,
                        "actual" : row["cmd"],
                        "prediction" : None, 
                        "message" : msg,
                        "raw" : None,
                        "error": str(e),
                        }

            preds.append(row_dict)
    return preds

