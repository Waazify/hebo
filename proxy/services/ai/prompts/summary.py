SUMMARY_PROMPT = """Given the following conversation happened between you and a user, generate a detailed summary in its original language as a single paragraph. Keep it straightforward, short, and concise.

For example, if the conversation below is in English, produce the summary in English.
If it is in Spanish, produce the summary in Spanish.
If the conversation is in Malaysian, produce the summary in Malaysian.
And so on.

Conversation
{conversation}
"""


def get_summary_prompt(conversation: str):
    return SUMMARY_PROMPT.format(
        conversation=conversation,
    )
