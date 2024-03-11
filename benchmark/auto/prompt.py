SYNTHETIC_QUERY_SYSTEM = '''You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.''' # system prompt

SYNTHETIC_QUERY_FEW_SHOT = '''Document: {document}
Question: {question}''' # few-shot prompt

SYNTHETIC_QUERY_USER = '''{few_shot_cases}
Document: {document}
Question: ''' # user prompt

