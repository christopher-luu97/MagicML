import os
import openai
import json
import sys

def get_key() -> str:
    """_summary_

    Returns:
        _type_: _description_
    """
    base_dir = os.path.dirname(sys.argv[0])
    creds_dir = os.path.join(base_dir, "config", "secrets.json")
    with open(creds_dir, "r") as f:
        config = json.load(f)
        return config['oai_credentials']


def get_response(input_str):
    """
    Get response from Open AI
    :param input_str:
    :return:
    """
    print("yo")
    __key = get_key()

    openai.api_key = __key

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=input_str,
        temperature=0.9,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    choices = response.get("choices")

    print(choices)

    choice_list = [choice.get("text").lstrip("\n") for choice in choices]

    return choice_list
