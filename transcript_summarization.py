# %%
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

load_dotenv()
# os.environ["OPENAI_API_KEY"] = your_key_here

with open("transcript_medium.txt", encoding="utf-8") as f:
    transcript = f.read()

# transcript[:1000]

# %% initialize LLM
chat_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", request_timeout=600)
chat_llm.get_num_tokens(transcript)  # 10,202 tokens; exceeds gpt-3.5's max 4,096 tokens

# %% try both NLTK and Recursive splitters, then assess
# https://python.langchain.com/en/latest/modules/indexes/text_splitters.html

# NLTK
nltk_splitter = NLTKTextSplitter(chunk_size=5000, chunk_overlap=500)
nltk_texts = nltk_splitter.split_text(transcript)
print(nltk_texts[0])
len(nltk_texts)

# RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "",  # https://github.com/hwchase17/langchain/issues/1663#issuecomment-1469161790
    ],
    chunk_size=5000,
    chunk_overlap=500,
)
recursive_texts = recursive_splitter.split_text(transcript)
print(recursive_texts[0])
len(recursive_texts)

# will go with NLTK as sentence structure is maintained when text is split into chunks
# %% create docs
docs = nltk_splitter.create_documents([transcript])
docs[0]
num_docs = len(docs)
num_tokens_first_doc = chat_llm.get_num_tokens(docs[0].page_content)
print(
    f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens"
)

# %% custom summarization prompt
map_prompt = """
The following is part of an intimate conversation between two individuals:
"{text}"
You are tasked to take notes - these will be used as teaching materials.
Document any major life events, career and achievements, beliefs and values, 
impact and legacy, personal traits and characteristics, and any other information that
would be essential when creating these lessons. 
These notes should be detailed and include relevant quotes where possible (they
have to be full sentences).
Exclude any mention of sponsors.
Do not hallucinate or make up any information.
NOTES AND QUOTES:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = """
You are an expert in writing biographies and capturing the motivations, emotions, 
successes, failures, and everyday realities of individuals.
From the following text delimited by triple backquotes, write a concise summary that 
contains multiple sections. 
```{text}```
Each section is an insightful life lesson. If lessons are similar, combine the sections.
Supplement each section with quotes that best captures the essence of the lesson.
Once all sections are complete, combine sections based on lesson similarity and overlaps. 
Rank each section for its depth, transformative potential and practicality. Sections 
related to personal growth and career development are ranked higher.
From this ranking, select the top 10 sections and discard the rest. 
Don't include the final ranking in the summary.
Use bullet points for each section.
Use a tone that reflects empathy, authenticity, understanding, and encouragement.
Do not hallucinate or make up any information.
CONCISE SUMMARY WITH MAXIMUM 10 SECTIONS:
"""

combine_prompt_template = PromptTemplate(
    template=combine_prompt, input_variables=["text"]
)

custom_summary_chain = load_summarize_chain(
    llm=chat_llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
    verbose=False,
)

custom_output = custom_summary_chain.run(docs)
print(custom_output)

# %%
