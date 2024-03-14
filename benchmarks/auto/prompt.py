SYNTHETIC_QUERY_SYSTEM = '''You are an expert question-answering system. You must create a question for the provided document. The question must be answerable within the context of the document.''' # system prompt

SYNTHETIC_QUERY_FEW_SHOT = '''Document: {document}
Question: {question}

''' # few-shot

SYNTHETIC_QUERY_USER = '''{few_shot_cases}Document: {document}
Question: ''' # user prompt

SYNTHETIC_ANSWER_SYSTEM='''You are a helpful assistant that are good at helping to answer a query based on the context step by step, the context is a document. If there is a good answer from the context, try to summarize the context as the answer. If the query doesn't form a complete question, or you don't know the answer, or there is no enough information to determine the answer, or the context is irrelevant to the question, just say I DON'T NO.''' # system prompt

SYNTHETIC_ANSWER_USER = '''Here is the question {question}
Here is the context: {document}''' # user prompt