# -*- coding: utf-8 -*-
"""All prompts used for fact-checking subtasks prompting."""

# Most of the prompt in this file comes from the Factcheck-GPT project:
# https://github.com/yuxiaw/factcheck-gpt

# updates in Dec are donimant of function-based or code-based prompts,
# to get rid of parsing LLM results
# ------------------------------------------------------------------------
# Dec 2023: decompose and decontextualise, return a list
# ------------------------------------------------------------------------
DOC_TO_INDEPEDENT_SENTENCES_PROMPT = """
Your task is to perform sentence segmentation and de-contextualization.
Let's define a function named process(input:str).
The return value should be a list of strings, where each string should be a decontextualized sentence.
For example, if a user call process("Mary is a five-year old girl. She likes playing piano. She doesn't like cookies.").
You should return a python list without any other words,
["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

process("{doc}")
"""


SENTENCES_TO_CLAIMS_PROMPT = """
Your task is to decompose the text into atomic claims.
Let's define a function named decompose(input:str).
The returned value should be a list of strings, where each string should be a context-independent claim, representing one fact.
For example, if a user call decompose("Mary is a five-year old girl, she likes playing piano and she doesn't like cookies.").
You should return a python list without any other words:
["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

decompose("{doc}")
"""

# just sentence splits without decontextualization
DOC_TO_SENTENCES_PROMPT = """
Your task is to perform sentence segmentation.
Let's define a function named split(input:str).
The return value should be a list of strings, where each string should be a sentence.
For example, if a user call process("Mary is a five-year old girl. She likes playing piano. She doesn't like cookies.").
You should return a python list without any other words,
["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

split("{doc}")
"""

# ------------------------------------------------------------------------
# Dec 2023: identify checkworthy
# ------------------------------------------------------------------------
CHECKWORTHY_PROMPT = """
Your task is to identify whether texts are checkworthy in the context of fact-checking.
Let's define a function named checkworthy(input: List[str]).
The return value should be a list of strings, where each string selects from ["Yes", "No"].
"Yes" means the text is a factual checkworthy statement.
"No" means that the text is not checkworthy, it might be an opinion, a question, or others.
For example, if a user call checkworthy(["I think Apple is a good company.", "Friends is a great TV series.", "Are you sure Preslav is a professor in MBZUAI?", "The Stanford Prison Experiment was conducted in the basement of Encina Hall.", "As a language model, I can't provide these info."])
You should return a python list without any other words,
["No", "Yes", "No", "Yes", "No"]
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!


checkworthy({texts})
"""

SPECIFY_CHECKWORTHY_CATEGORY_PROMPT = """
You are a factchecker assistant with task to identify a sentence, whether it is 1. a factual claim; 2. an opinion; 3. not a claim (like a question or a imperative sentence); 4. other categories.
Let's define a function named checkworthy(input: str).
The return value should be a python int without any other words, representing index label, where index selects from [1, 2, 3, 4].

For example, if a user call checkworthy("I think Apple is a good company.")
You should return 2
If a user call checkworthy("Friends is a great TV series.")
You should return 1
If a user call checkworthy("Are you sure Preslav is a professor in MBZUAI?")
You should return 3
If a user call checkworthy("As a language model, I can't provide these info.")
You should return 4
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

checkworthy("{sentence}")
"""


# ------------------------------------------------------------------------
# Dec 2023: Verify
# ------------------------------------------------------------------------
IDENTIFY_STANCE_PROMPT = """You are given a claim and an evidence text, and you need to decide whether the evidence supports, refutes, or is irrelevant to the claim. Choose from the following three options.
A. The evidence supports the claim.
B. The evidence refutes the claim.
C. The evidence is irrelevant to the claim.

For example, you are give Claim: "Preslav is a professor.", Evidence: "Preslav Nakov is a Professor in MBZUAI NLP group, and also the department chair." You should return A
Pick the correct option A, B, C without other words.

Claim: {claim}
Evidence: {evidence}"""


VERIFY_PROMPT = """
You are given a piece of text. Your task is to identify whether there are any factual errors within the text.
When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences may be helpful. Some evidences may contradict to each other. You must be careful when using the evidences to judge the factuality of the given text.
The response should be a dictionary with three keys - "reasoning", "factuality", "error", and "correction", which correspond to the reasoning, whether the given text is factual or not (Boolean - True or False), the factual error present in the text, and the corrected text.
The following is the given text
[text]: {claim}
The following is the provided evidences
[evidences]: {evidence}
You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPONSE WITH '{{'.
[response format]:
{{
    "reasoning": "Why is the given text factual or non-factual? Be careful when you said something is non-factual. When you said something is non-factual, you must provide multiple evidences to support your decision.",
    "error": "None if the text is factual; otherwise, describe the error.",
    "correction": "The corrected text if there is an error.",
    "factuality": True if the given text is factual, False otherwise.
}}
"""
# ------------------------------------------
# Oct 2023
# ------------------------------------------
zero_shot_sentence_checkworthiness = """You are a factchecker assistant with task to identify sentences that are checkworthy. Sentence is checkworthy only if it contains factual claims.
Classify the check-worthiness of these sentences, and output the label yes or no:
{sentence}
output:
"""

zero_shot_claim_checkworthiness = """You are a factchecker assistant with task to identify a sentence, whether it is 1. a factual claim; 2. an opinion; 3. not a claim (like a question or a imperative sentence); 4. other categories. \n
Output the label index only: \n
{claim} \n
output:
"""

# We find that it is hard for model to distinguish complete support and partial support, merge as the one: support
zero_shot_claim_evidence_three_stance_prompt = "### Instruction: You are given a claim and an evidence text, and you need to decide whether the evidence supports, refutes, or is irrelevant to the claim.\n\n### Input:\n\nClaim: {claim}\n\nEvidence: {evidence}\n\nOptions are as follows:\n A) The evidence supports the claim.\n\n B) The evidence refutes the claim.\n C) The evidence is irrelevant to the claim.\n\n Pick the correct option. \n\n### Final Answer: "


zero_shot_claim_evidence_stance = """Given the evidence \n {evidence}, determine if the following statement is completely supported, partially supported, refuted or is irrelevant: {claim}, choose from four labels: 1. completely support, 2. partially support, 3. refute and 4. irrelevant.
Return the label index only.
Label index:
"""

zero_shot_nli = """Given the premise sentence {}, determine if the following statement is entailed or contradicted or neutral: {}, by three labels: entailment, contradiction, neutral.
Label:
"""


zero_shot_edit_response = """Given a document containing factual errors, please correct the errors in the document depending on a corresponding list of factually true claims. Note that preserve the linguistic features and style of the original document, just correct factual errors.

document: {response}

true claims: {claims}

revised document: """

zero_shot_edit_response_given_question = """Given a question, and an answer containing factual errors, please correct the errors in the document depending on a corresponding list of factually true claims. Note that preserve the linguistic features and style of the original document, just correct factual errors.

question: {prompt}

document: {response}

true claims: {claims}

revised document: """

# -------------------------------------------------------------------
# July 2023: decompose and decontextualise into atomic claims
# -------------------------------------------------------------------
# ZERO_SHOT_SENTENCE_TO_ATOMIC_CLAIMS = """Depending the context: {}, please breakdown the following sentence into independent facts and replace pronouns such as it, they, those, these, this, that, with specific entities or events.
# The sentence is: {}
# Atomic facts for this sentence are: """

ZERO_SHOT_SENTENCE_TO_ATOMIC_CLAIMS = """Depending the context: {}, please breakdown the following sentence into independent facts.
The sentence is: {}
Atomic facts for this sentence are: """


FEW_SHOT_SENTENCE_TO_ATOMIC_CLAIMS = """Depending the context, please breakdown the following sentence into independent facts.

Context: The United States has had two black presidents: Barack Obama, who served two terms from 2009 to 2017, and Donald Trump, who served one term from 2017 to 2021. Obama was the first black president in the history of the United States. He was born in Honolulu, Hawaii, to a mother from Kansas and a father from Kenya. Trump was the second black president. He was born in New York City and previously served as a businessman and reality television personality.

The sentence is: The United States has had two black presidents: Barack Obama, who served two terms from 2009 to 2017, and Donald Trump, who served one term from 2017 to 2021.
Atomic facts for this sentence are:
[
    "The United States has had two black presidents: Barack Obama and Donald Trump.",
    "Black president Barack Obama served two terms from 2009 to 2017.",
    "Black president Donald Trump served one term from 2017 to 2021."
]

The sentence is: Obama was the first black president in the history of the United States.
Atomic facts for this sentence are:
[
    "Obama was the first black president in the history of the United States."
]

The sentence is: He was born in Honolulu, Hawaii, to a mother from Kansas and a father from Kenya.
Atomic facts for this sentence are:
[
    "Barack Obama was born in Honolulu, Hawaii.",
    "Barack Obama mother was from Kansas.",
    "Barack Obama father was from Kenya."
]

The sentence is: Trump was the second black president.
Atomic facts for this sentence are:
[
    "Trump was the second black president."
]

The sentence is: He was born in New York City and previously served as a businessman and reality television personality.
Atomic facts for this sentence are:
[
    "Donald Trump was born in New York City.",
    "Donald Trump previously served as a businessman",
    "Donald Trump previously served as a reality television personality."
]


Context: In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas. He was born on October 16, 1898, and served on the Supreme Court from 1939 until his retirement in 1975. Therefore, in 1980, Justice Douglas was still alive and would have been the oldest serving justice on the Court at that time.
The sentence is: In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas.
Atomic facts for this sentence are:
[
    "In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas."
]

The sentence is: He was born on October 16, 1898, and served on the Supreme Court from 1939 until his retirement in 1975.
Atomic facts for this sentence are:
[
    "Justice William O. Douglas was born on October 16, 1898."
    "Justice William O. Douglas served on the Supreme Court from 1939 until his retirement in 1975."
]

The sentence is: Therefore, in 1980, Justice Douglas was still alive and would have been the oldest serving justice on the Court at that time.
Atomic facts for this sentence are:
[
    "Therefore, in 1980, Justice Douglas was still alive."
    "Justice William O. Douglas would have been the oldest serving justice on the Court in 1980."
]


Context: There have been only four female presidents of the United States in the country's history, so it is difficult to determine an average height for this group. The four female presidents were: \r\n1.Abigail Adams (1797-1801) \r\n2.Marilyn Carlson Nelson (2009-2013) \r\n3.Luci Baines Johnson (1973-1977) \r\n4.Hillary Clinton (2017-2021)
The sentence is: There have been only four female presidents of the United States in the country's history, so it is difficult to determine an average height for this group.
Atomic facts for this sentence are:
[
    "There have been only four female presidents of the United States in the country's history.",
    "It is difficult to determine an average height for four female presidents of the United States."
]

The sentence is: The four female presidents were: \r\n1.Abigail Adams (1797-1801) \r\n2.Marilyn Carlson Nelson (2009-2013) \r\n3.Luci Baines Johnson (1973-1977) \r\n4.Hillary Clinton (2017-2021)
Atomic facts for this sentence are:
[
    "Abigail Adams (1797-1801) is a female president of the United States.",
    "Marilyn Carlson Nelson (2009-2013) is a female president of the United States.",
    "Luci Baines Johnson (1973-1977) is a female president of the United States.",
    "Hillary Clinton (2017-2021) is a female president of the United States."
]


Context: {}
The sentence is: {}
Atomic facts for this sentence are:
"""

# This prompt aims to break the document into decontextualised sentences, and then atomic claims
# Though it can not decontexlualize sentences, it can better break all sentences than the prompt above
# combined with using system_role = "You are good at document decomposition and decontextualization."
# date: 22/10/2023
FEW_SHOT_DECONTEXTUALIZE_SENTENCE_ATOMIC_CLAIMS = """Depending the context, please break it down into independent sentences, and breakdown the sentence into independent facts.
Context: The United States has had two black presidents: Barack Obama, who served two terms from 2009 to 2017, and Donald Trump, who served one term from 2017 to 2021. Obama was the first black president in the history of the United States. He was born in Honolulu, Hawaii, to a mother from Kansas and a father from Kenya. Trump was the second black president. He was born in New York City and previously served as a businessman and reality television personality.

The sentence is: The United States has had two black presidents: Barack Obama, who served two terms from 2009 to 2017, and Donald Trump, who served one term from 2017 to 2021.
Atomic facts for this sentence are:
[
    "The United States has had two black presidents: Barack Obama and Donald Trump.",
    "Black president Barack Obama served two terms from 2009 to 2017.",
    "Black president Donald Trump served one term from 2017 to 2021."
]

The sentence is: Obama was the first black president in the history of the United States.
Atomic facts for this sentence are:
[
    "Obama was the first black president in the history of the United States."
]

The sentence is: Barack Obama was born in Honolulu, Hawaii, to a mother from Kansas and a father from Kenya.
Atomic facts for this sentence are:
[
    "Barack Obama was born in Honolulu, Hawaii.",
    "Barack Obama mother was from Kansas.",
    "Barack Obama father was from Kenya."
]

The sentence is: Trump was the second black president.
Atomic facts for this sentence are:
[
    "Trump was the second black president."
]

The sentence is: Donald Trump was born in New York City and previously served as a businessman and reality television personality.
Atomic facts for this sentence are:
[
    "Donald Trump was born in New York City.",
    "Donald Trump previously served as a businessman",
    "Donald Trump previously served as a reality television personality."
]


Context: In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas. He was born on October 16, 1898, and served on the Supreme Court from 1939 until his retirement in 1975. Therefore, in 1980, Justice Douglas was still alive and would have been the oldest serving justice on the Court at that time.
The sentence is: In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas.
Atomic facts for this sentence are:
[
    "In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas."
]

The sentence is: Justice William O. Douglas was born on October 16, 1898, and served on the Supreme Court from 1939 until his retirement in 1975.
Atomic facts for this sentence are:
[
    "Justice William O. Douglas was born on October 16, 1898."
    "Justice William O. Douglas served on the Supreme Court from 1939 until his retirement in 1975."
]

The sentence is: Therefore, in 1980, Justice Douglas was still alive and would have been the oldest serving justice on the Court at that time.
Atomic facts for this sentence are:
[
    "Therefore, in 1980, Justice Douglas was still alive."
    "Justice William O. Douglas would have been the oldest serving justice on the Court in 1980."
]


Context: There have been only four female presidents of the United States in the country's history, so it is difficult to determine an average height for this group. The four female presidents were: \r\n1.Abigail Adams (1797-1801) \r\n2.Marilyn Carlson Nelson (2009-2013) \r\n3.Luci Baines Johnson (1973-1977) \r\n4.Hillary Clinton (2017-2021)
The sentence is: There have been only four female presidents of the United States in the country's history, so it is difficult to determine an average height for this group.
Atomic facts for this sentence are:
[
    "There have been only four female presidents of the United States in the country's history.",
    "It is difficult to determine an average height for four female presidents of the United States."
]

The sentence is: The four female presidents were: \r\n1.Abigail Adams (1797-1801) \r\n2.Marilyn Carlson Nelson (2009-2013) \r\n3.Luci Baines Johnson (1973-1977) \r\n4.Hillary Clinton (2017-2021)
Atomic facts for this sentence are:
[
    "Abigail Adams (1797-1801) is a female president of the United States.",
    "Marilyn Carlson Nelson (2009-2013) is a female president of the United States.",
    "Luci Baines Johnson (1973-1977) is a female president of the United States.",
    "Hillary Clinton (2017-2021) is a female president of the United States."
]


Context: {}
The sentence is: {}
Atomic facts for this sentence are:
"""


# -------------------------------------------------------------------
# April 2023: overall simple pipeline prompts
# -------------------------------------------------------------------
DECONTEXTILISATION_PROMPT = """Decompose and decontextualise a document into independently meaningful sentences. This process will make each sentence stand alone that can be verified independently.

Input: Mary is a five-year old girl. She likes playing piano. She doesn't like cookies.
Output:
Mary is a five-year old girl.
Mary likes playing piano.
Mary doesn't like cookies.

Input: Google began as an online search firm, but it now offers more than 50 Internet services and products, from e-mail and online document creation to software for mobile phones and tablet computers. In addition, its 2012 acquisition of Motorola Mobility put it in the position to sell hardware in the form of mobile phones.
Ouput:
Google began as an online search firm.
Google now offers more than 50 Internet services and products.
Google offers from e-mail and online document creation to software for mobile phones and tablet computers.
Google 2012 acquisition of Motorola Mobility put it in the position to sell hardware in the form of mobile phones.

Input: """


CHECK_WORTHINESS_LABEL_ONLY_PROMPT = """Identify whether this claim is an opinion or factual, and whether it is checkworthy or not in the context of fact-checking. Just return two labels without explanation.
I think Apple is a good company.
opinon, not checkworthy
Preslav is a professor in MBZUAI.
factual, checkworthy
Friends is a great TV series.
opinion, not checkworthy
The Stanford Prison Experiment was conducted in the basement of Encina Hall.
factual, checkworthy
"""


ENTITY_EXTRACTION_PROMPT = """Extract all entities of a claim.
Input: Google now offers more than 50 Internet services and products.
Output: Google, Internet services, product
Input: Donald John Trump is an American politician, media personality, and businessman.
Output: Donald John Trump, American politician, media personality, businessman
Input: """


QGEN_PROMPT = """Give a list of queries using for searching related information for a claim.
Input: Google now offers more than 50 Internet services and products.
Output: What does Google offers now?
How many service and product does Google offer?
Google, more than 50 Internet services, products
Input: Donald John Trump is an American politician, media personality, and businessman.
Output: Who is Donald John Trump?
Give information of Donald John Trump.
Donald John Trump, American politician
Donald John Trump, media personality
Donald John Trump, businessman
Input: """


STANCE_DETECTION_PROMPT = """Determine whether the evidence support the claim or not. Choose label from [support, partial support, refute, other] and explain why.
Support means we can entail the claim by the evidence.
Partial support means: part of the information presented in the claim appear in the evidence.
Refute means that the evidence mention the same event as the claim, but a clear opposite fact. It should be highlighed that under refute, the evidence mentions the fact in the claim, they are closely relevant, but opposite meaning or stance.
Other means the evidence does not mention anything about the fact described in the claim, such that it neither supports nor refutes the claim.

Claim: Elon Musk is the founder, CEO and chief engineer of SpaceX.
Evidence: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Stance: support, statement 'he is also the founder, CEO and chief engineer of SpaceX' in evidence above supports the claim.

Claim: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Evidence: Elon Musk is the founder, CEO and chief engineer of SpaceX.
Stance: partial support.

Claim: Steve Jobs is the founder, CEO and chief engineer of SpaceX.
Evidence: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Stance: refute.

Claim: Elon Musk is a professor in The Stanford University.
Evidence: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Stance: other, according to the evidence, I cannot judge whether the claim is true or not, not enough information, the evidence neither supports nor refutes.

Claim: On January 6, 2021, a mob of supporters of former President Donald Trump stormed the U.S. Capitol in an attempt to overturn the 2020 presidential election.
Evidence: On January 6, 2021, following the defeat of U.S. President Donald Trump in the 2020 presidential election, a mob of his supporters attacked the United States Capitol Building in Washington, D.C. The mob sought to keep Trump in power by preventing a joint session of Congress from counting the electoral college votes to formalize the victory of President-elect Joe Biden.
Stance: support.

Claim: The 2021 Capitol Hill riots resulted in the deaths of five people, including a Capitol police officer.
Evidence: Five people died either shortly before, during, or following the riot: one was shot by Capitol Police, another died of a drug overdose, and three died of natural causes.
Stance: partial support, the evidence supports that fact that five deaths, but not sure whether they include a Capitol police officer or not.

Claim: More than 300 people have been charged with crimes related to the riots.
Evidence: As of November 10, 2022, over 940 people had been charged in the Capitol breach.
Stance: refute, evidence and claim are describing the same thing, the number of people who was charged is over 940, while more than 300 in the claim, so the evidence refutes the claim.

Claim: More than 300 people have been charged with crimes related to the riots.
Evidence: The laptop computer taken from Pelosi's office was taken by 22-year-old Capitol rioter Riley Williams. Williams was arrested and indicted on eight counts, including theft of government property, obstructing an official proceeding, and assaulting or resisting police.
Stance: other, the evidence demonstrates something relevent to the fact in the claim, but it does not support or refute any information of it.

Claim: {}
Evidence: {}
Stance: """


EDITOR_PROMPT = """Fix the claim according to the evidence.

Claim: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
Evidence: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
This suggests 45 minutes switch time in your statement is wrong.
Fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle.

Claim: In the battles of Lexington and Concord, the British side was led by General Thomas Hall.
Evidence: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
This suggests General Thomas Hall in your statement is wrong.
Fix: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

Claim: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
Evidence: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
This suggests Encina Hall in your statement is wrong.
Fix: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.

Claim: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
Evidence: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
This suggests the Havel-Hakimi algorithm’s functionality in your statement is wrong.
Fix: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi.

Claim: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Phil Ramone.
Evidence: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
This suggests "Time of My Life" producer name in your statement is wrong.
Fix: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.

Claim: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 1.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.
Evidence: Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.
This suggests the 1.4 million square feet of built-up space in your statment is wrong.
Fix: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

Claim: {claim}
Evidence: {evidence}
This suggests
""".strip()

# this prompt is generated by FBzzh for test
REJECT_RATE_PROMPT = """You will receive a question and an evidence,
determine whether you can answer the question based on the given evidence.
If you can not answer the question, output 'sorry, cannot answer the question',
else, output your answer.

Question: {question}
Evidence: {evidence}
Answer: 
"""