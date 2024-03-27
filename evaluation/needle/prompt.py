
file_list_pt = ['./evaluation/needle/PaulGrahamEssays/apple.pt', 
'./evaluation/needle/PaulGrahamEssays/submarine.pt', 
'./evaluation/needle/PaulGrahamEssays/addiction.pt', 
'./evaluation/needle/PaulGrahamEssays/gap.pt', 
'./evaluation/needle/PaulGrahamEssays/bias.pt', 
'./evaluation/needle/PaulGrahamEssays/useful.pt', 
'./evaluation/needle/PaulGrahamEssays/popular.pt', 
'./evaluation/needle/PaulGrahamEssays/vcsqueeze.pt', 
'./evaluation/needle/PaulGrahamEssays/gba.pt', 
'./evaluation/needle/PaulGrahamEssays/island.pt', 
'./evaluation/needle/PaulGrahamEssays/before.pt', 
'./evaluation/needle/PaulGrahamEssays/todo.pt', 
'./evaluation/needle/PaulGrahamEssays/vb.pt', 
'./evaluation/needle/PaulGrahamEssays/founders.pt', 
'./evaluation/needle/PaulGrahamEssays/unions.pt', 
'./evaluation/needle/PaulGrahamEssays/diff.pt', 
'./evaluation/needle/PaulGrahamEssays/vw.pt', 
'./evaluation/needle/PaulGrahamEssays/corpdev.pt', 
'./evaluation/needle/PaulGrahamEssays/wisdom.pt', 
'./evaluation/needle/PaulGrahamEssays/love.pt', 
'./evaluation/needle/PaulGrahamEssays/sun.pt', 
'./evaluation/needle/PaulGrahamEssays/langdes.pt', 
'./evaluation/needle/PaulGrahamEssays/pow.pt', 
'./evaluation/needle/PaulGrahamEssays/nft.pt', 
'./evaluation/needle/PaulGrahamEssays/laundry.pt', 
'./evaluation/needle/PaulGrahamEssays/weird.pt', 
'./evaluation/needle/PaulGrahamEssays/siliconvalley.pt', 
'./evaluation/needle/PaulGrahamEssays/worked.pt', 
'./evaluation/needle/PaulGrahamEssays/rootsoflisp.pt', 
'./evaluation/needle/PaulGrahamEssays/goodtaste.pt', 
'./evaluation/needle/PaulGrahamEssays/copy.pt', 
'./evaluation/needle/PaulGrahamEssays/want.pt', 
'./evaluation/needle/PaulGrahamEssays/desres.pt', 
'./evaluation/needle/PaulGrahamEssays/know.pt', 
'./evaluation/needle/PaulGrahamEssays/hubs.pt', 
'./evaluation/needle/PaulGrahamEssays/iflisp.pt', 
'./evaluation/needle/PaulGrahamEssays/foundervisa.pt', 
'./evaluation/needle/PaulGrahamEssays/superangels.pt', 
'./evaluation/needle/PaulGrahamEssays/boss.pt', 
'./evaluation/needle/PaulGrahamEssays/aord.pt', 
'./evaluation/needle/PaulGrahamEssays/newideas.pt', 
'./evaluation/needle/PaulGrahamEssays/avg.pt', 
'./evaluation/needle/PaulGrahamEssays/gh.pt', 
'./evaluation/needle/PaulGrahamEssays/rss.pt', 
'./evaluation/needle/PaulGrahamEssays/startuplessons.pt', 
'./evaluation/needle/PaulGrahamEssays/philosophy.pt', 
'./evaluation/needle/PaulGrahamEssays/ecw.pt', 
'./evaluation/needle/PaulGrahamEssays/mod.pt', 
'./evaluation/needle/PaulGrahamEssays/web20.pt']






file_list = ['./evaluation/needle/PaulGrahamEssays/apple.txt', './evaluation/needle/PaulGrahamEssays/submarine.txt', './evaluation/needle/PaulGrahamEssays/addiction.txt', './evaluation/needle/PaulGrahamEssays/gap.txt', './evaluation/needle/PaulGrahamEssays/bias.txt', './evaluation/needle/PaulGrahamEssays/useful.txt', './evaluation/needle/PaulGrahamEssays/popular.txt', './evaluation/needle/PaulGrahamEssays/vcsqueeze.txt', './evaluation/needle/PaulGrahamEssays/gba.txt', './evaluation/needle/PaulGrahamEssays/island.txt', './evaluation/needle/PaulGrahamEssays/before.txt', './evaluation/needle/PaulGrahamEssays/todo.txt', './evaluation/needle/PaulGrahamEssays/vb.txt', './evaluation/needle/PaulGrahamEssays/founders.txt', './evaluation/needle/PaulGrahamEssays/unions.txt', './evaluation/needle/PaulGrahamEssays/diff.txt', './evaluation/needle/PaulGrahamEssays/vw.txt', './evaluation/needle/PaulGrahamEssays/corpdev.txt', './evaluation/needle/PaulGrahamEssays/wisdom.txt', './evaluation/needle/PaulGrahamEssays/love.txt', './evaluation/needle/PaulGrahamEssays/sun.txt', './evaluation/needle/PaulGrahamEssays/langdes.txt', './evaluation/needle/PaulGrahamEssays/pow.txt', './evaluation/needle/PaulGrahamEssays/nft.txt', './evaluation/needle/PaulGrahamEssays/laundry.txt', './evaluation/needle/PaulGrahamEssays/weird.txt', './evaluation/needle/PaulGrahamEssays/siliconvalley.txt', './evaluation/needle/PaulGrahamEssays/worked.txt', './evaluation/needle/PaulGrahamEssays/rootsoflisp.txt', './evaluation/needle/PaulGrahamEssays/goodtaste.txt', './evaluation/needle/PaulGrahamEssays/copy.txt', './evaluation/needle/PaulGrahamEssays/want.txt', './evaluation/needle/PaulGrahamEssays/desres.txt', './evaluation/needle/PaulGrahamEssays/know.txt', './evaluation/needle/PaulGrahamEssays/hubs.txt', './evaluation/needle/PaulGrahamEssays/iflisp.txt', './evaluation/needle/PaulGrahamEssays/foundervisa.txt', './evaluation/needle/PaulGrahamEssays/superangels.txt', './evaluation/needle/PaulGrahamEssays/boss.txt', './evaluation/needle/PaulGrahamEssays/aord.txt', './evaluation/needle/PaulGrahamEssays/newideas.txt', './evaluation/needle/PaulGrahamEssays/avg.txt', './evaluation/needle/PaulGrahamEssays/gh.txt', './evaluation/needle/PaulGrahamEssays/rss.txt', './evaluation/needle/PaulGrahamEssays/startuplessons.txt', './evaluation/needle/PaulGrahamEssays/philosophy.txt', './evaluation/needle/PaulGrahamEssays/ecw.txt', './evaluation/needle/PaulGrahamEssays/mod.txt', './evaluation/needle/PaulGrahamEssays/web20.txt']

ANTHROPIC_TEMPLATE_REV1 = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct

Human: <context>
{context}
</context>

{question} Don't give information outside the document or repeat your findings. Respond with "unanswerable" if the information is not available in the context.

Assistant: Here is the most relevant sentence in the context:
'''

ANTHROPIC_TEMPLATE_REV1_ED = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct

Human: <context>
{context}
</context>

{question} Respond with "unanswerable" if the information is not available in the context. Don't give information outside the document or repeat your findings. Don't repeat your findings. Assistant: Here is the most relevant sentence in the context:
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

ANTHROPIC_TEMPLATE_ORIGINAL_LIST = [  
    "Human: You are a close-reading bot with a great memory who answers questions for users. I'm going to give you the text of some essays. Amidst these essays (\"the haystack\") I've inserted a sentence (\"the needle\") that contains an answer to the user's question. Here's the question:\n<question>",   
    "question",  
    "</question>\nHere's the text of the essays. The answer appears in it somewhere.\n<haystack>",  
    "context",  
    "</haystack>\nNow that you've read the context, please answer the user's question, repeated one more time for ease of reference:\n<question>",  
    "question",  
    "</question>\nTo do so, first find the sentence from the haystack that contains the answer (there is such a sentence, I promise!) and put it inside <most_relevant_sentence> XML tags. Then, put your answer in <answer> tags. Base your answer strictly on the context, without reference to outside information. Thank you.\nIf you can't find the answer return the single word UNANSWERABLE.\nAssistant:"  
]  



SIMPLE_TEMPLATE = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct.
The following is a set of context and a question that will relate to the context. 
#CONTEXT
{context}
#ENDCONTEXT

#QUESTION
{question} Don't give information outside the document or repeat your findings. If the information is not available in the context respond UNANSWERABLE.
'''
#The leading spaces in template make a difference so removed them
GEMINI_TEMPLATE = '''
You are a helpful AI bot that answers questions for a user. Keep your response short and direct.
The following is a set of context and a question that will relate to the context. 
#CONTEXT
{context}
#ENDCONTEXT

#QUESTION
{question} Don't give information outside the document or repeat your findings. If the information is not available in the context respond UNANSWERABLE.'''

GEMINI_TEMPLATE2 = '''
<context>
{context}
</context>
{question} Don't give information outside the document or repeat your findings. 
Here is the magic number from the context:

'''