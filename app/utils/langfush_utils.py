import os

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

load_dotenv()

LANGFUSE_PUBLIC_KEY = os.getenv("langfuse_public_key")
LANGFUSE_SECRET_KEY = os.getenv("langfuse_secret_key")
LANGFUSE_HOST = os.getenv("langfuse_host")

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST
)

langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST
)


def upload_prompt(
    prompt_name, prompt_description, prompt_labels, prompt_config, prompt_json_schema
):
    """
    Example usage:
    prompt_name="tool selection prompt",
    prompt_description=Prompt,
    prompt_labels=["stage_v2"],
    prompt_config={"model": "gpt-4.1-2025-04-14", "temperature": 0, "json_schema": {}},
    prompt_json_schema=""
    """
    langfuse.create_prompt(
        name=prompt_name,
        prompt=prompt_description,
        config=prompt_config,
        labels=prompt_labels,
    )


def config_llm_callback(run_name, tag, conversation_id, user_id):
    config = {
        "callbacks": [langfuse_handler],
        "run_name": "journey_action_detection",
        "tags": [tag],
        "metadata": {
            "langfuse_session_id": conversation_id,
            "langfuse_user_id": user_id,
        },
    }
    return config


def get_prompt_config(prompt_tag, label):
    prompt_ = langfuse.get_prompt(prompt_tag, label=label)
    llm_config = prompt_.config
    prompt = prompt_.compile()
    return prompt, llm_config
