CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History
{chat_history}

Follow up question
{standalone_question}
"""


def get_condense_prompt(chat_history: str, standalone_question: str):
    return CONDENSE_PROMPT.format(
        chat_history=chat_history, standalone_question=standalone_question
    )
