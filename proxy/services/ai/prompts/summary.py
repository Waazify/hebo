SUMMARY_PROMPT = """Given the following conversation happened between you and a user, generate a detailed summary in English as a single paragraph. Keep it straightforward, short, and concise.
At the end of the summary, add a comment with the language of the conversation adding the following note: "Primary Language used in this conversation:" and then the language.

For example, if the conversation below is in Spanish, produce the following:
|Summary in English|
Primary Language used in this conversation: Spanish

Conversation
{conversation}
"""


def get_summary_prompt(conversation: str):
    return SUMMARY_PROMPT.format(
        conversation=conversation,
    )
