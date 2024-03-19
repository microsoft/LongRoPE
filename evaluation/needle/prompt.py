
ANTHROPIC_TEMPLATE_REV1 = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct

Human: <context>
{context}
</context>

{question} Don't give information outside the document or repeat your findings. Respond 
with "unanswerable" if the information is not available in the context.

Assistant: Here is the most relevant sentence in the context:
'''

ANTHROPIC_TEMPLATE_REV2 = '''
Human: You are a close-reading bot with a great memory who answers questions for users. I'm going to give you the text of some essays. Amidst these essays ("the haystack") I've inserted a sentence ("the needle") that contains an answer to the user's question. Here's the question:
<question>{question}</question>
Here's the text of the essays. The answer appears in it somewhere.
<haystack>
{context}
</haystack>
Now that you've read the context, please answer the user's question, repeated one more time for ease of reference:
<question>{question}</question>
To do so, first find the sentence from the haystack that contains the answer (there is such a sentence, I promise!) and put it inside <most_relevant_sentence> XML tags. Then, put your answer in <answer> tags. Base your answer strictly on the context, without reference to outside information. Thank you.
If you can't find the answer return the single word UNANSWERABLE.
Assistant: Here is the most relevant sentence in the context:'''

ANTHROPIC_TEMPLATE_ORIGINAL = '''Human: You are a close-reading bot with a great memory who answers questions for users. I'm going to give you the text of some essays. Amidst these essays ("the haystack") I've inserted a sentence ("the needle") that contains an answer to the user's question. Here's the question:
<question>{question}</question>
Here's the text of the essays. The answer appears in it somewhere.
<haystack>
{context}
</haystack>
Now that you've read the context, please answer the user's question, repeated one more time for ease of reference:
<question>{question}</question>

To do so, first find the sentence from the haystack that contains the answer (there is such a sentence, I promise!) and put it inside <most_relevant_sentence> XML tags. Then, put your answer in <answer> tags. Base your answer strictly on the context, without reference to outside information. Thank you.
If you can't find the answer return the single word UNANSWERABLE.
Assistant:'''

SIMPLE_TEMPLATE = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct.
The following is a set of context and a question that will relate to the context. 
#CONTEXT
{context}
#ENDCONTEXT

#QUESTION
{question} Don't give information outside the document or repeat your findings. If the
information is not available in the context respond UNANSWERABLE.
'''
#The leading spaces in template make a difference so removed them
GEMINI_TEMPLATE = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct.
The following is a set of context and a question that will relate to the context. 
#CONTEXT
{context}
#ENDCONTEXT

#QUESTION
{question} Don't give information outside the document or repeat your findings. If the
information is not available in the context respond UNANSWERABLE.'''

GEMINI_TEMPLATE2 = '''
<context>
{context}
</context>
{question} Don't give information outside the document or repeat your findings. 
Here is the magic number from the context:

'''