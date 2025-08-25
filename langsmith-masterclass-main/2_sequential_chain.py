from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Sequential Chain'

prompt1 = PromptTemplate(
    template="Generate a detailed report on '{topic}' in 100 words ",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI(temperature=0.1)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name':'sequential run',
    'configurable': {'thread_id': '1'}
}

result = chain.invoke({'topic': 'Unemployment in India'},config = config)

print(result)
