import json


def make_doc_prompt(doc, doc_id, doc_prompt, method=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    if method == "vanilla":
        text = doc['text']
    elif method == "summary":
        text = doc["summary"]
    elif method == "snippet":
        text = doc["extraction"]
    else:
        raise ValueError("Don't support such method.")
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, method=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    doc_texts = []
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "")  # if there is no doc we also delete the empty line
        else:
            doc_list = item["docs"][:ndoc]
            for doc_id, doc in enumerate(doc_list):
                doc_texts.append(make_doc_prompt(doc, doc_id, doc_prompt, method=method))
            text = "".join(doc_texts)
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip()  # remove any space or \n

    return prompt, doc_texts


def create_eli5_eval_dataset(result_path):
    eval_data = json.load(open(result_path, "r"))
    for data in eval_data:
        yield {
            "question": data["question"],
            "contexts": data["contexts"],
            "answers": data["output"],
            "gt_answers": data["claims"]
        }
