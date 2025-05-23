# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings 

embedding_function = OpenAIEmbeddings(model = 'text-embedding-3-large')

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name='income_tax_collection',
    persist_directory='./income_tax_collection'
)

retriver = vector_store.as_retriever(search_kwargs={'k': 3})

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document 
from langgraph.graph import StateGraph 

class AgentState(TypedDict): 
    query : str 
    context : List[Document]
    answer : str 

graph_builder = StateGraph(AgentState)

# %%
def retrive(state: AgentState) -> AgentState:
    query = state['query']
    docs = retriver.invoke(query)
    return {'context': docs}

# %%
from langchain_openai import ChatOpenAI 
llm = ChatOpenAI(model='gpt-4o')

# %%
from langchain import hub 

generate_prompt = hub.pull("rlm/rag-prompt")

generate_llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=100) 

def generate(state: AgentState) -> AgentState: 
    context = state['context']
    query = state['query']

    rag_chain = generate_prompt | generate_llm

    response = rag_chain.invoke({"question": query, 'context': context}) 

    return {'answer': response.content}

# %%
from langchain import hub
from typing import Literal

doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    """
    주어진 state를 기반으로 문서의 관련성을 판단합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        Literal['relevant', 'irrelevant']: 문서가 관련성이 높으면 'relevant', 그렇지 않으면 'irrelevant'를 반환합니다.
    """
    query = state['query']
    context = state['context']

    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})

    if response['Score'] == 1:
        # 2.3장과 다르게 `relevant`와 `irrelevant`를 반환합니다
        # node를 직접 지정하는 것보다 실제 판단 결과를 리턴하면서 해당 node의 재사용성을 높일 수 있습니다.
        return 'relevant'
    
    return 'irrelevant'

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요 
사전: {dictionary}                                           
질문: {{query}}
""")

def rewrite(state: AgentState) -> AgentState:
    """
    사용자의 질문을 사전을 참고하여 변경합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다.
    """
    query = state['query']
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query})
    return {'query': response}


# %%
from langchain import hub 

# hallucination_prompt = hub.pull("langchain-ai/rag-answer-hallucination")

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_halluciantion(state: AgentState) -> Literal['hallucinated', 'not hallucinated'] : 
# def check_halluciantion(state: AgentState) : 
    answer = state['answer']
    context = state['context'] 

    context = [doc.page_content for doc in context]

    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    # hallucination_chain = hallucination_prompt | hallucination_llm 
    
    response = hallucination_chain.invoke({
        'student_answer': answer, 
        'documents': context
    })

    return response 
    

# # %%
# query = "연봉 6천만원인 거주자의 소득세는?"
# retrive_context = {
#     'query' : query 
# }

# context = retrive(retrive_context)['context']

# generate_state = {
#     'query' : query,
#     'context': context 
# }

# answer = generate(generate_state)

# print(answer)

# hallucination_state = {
#     'answer' : answer, 
#     'context' : context 
# }

# check_halluciantion(hallucination_state)


# %%
helpfulness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState) -> Literal['helpful', 'unhelpful']: 
    query = state['query']
    answer = state['answer'] 

    helpfulness_chain = helpfulness_prompt | llm 

    response = helpfulness_chain.invoke({
        'question': query, 
        'student_answer': answer
    })

    if response['Score'] == 1:
        return 'helpful'

    return 'unhelpful'

def check_helpfulness(state: AgentState) -> AgentState: 
    return state 
    

# %%
query = "연봉 6천만원인 거주자의 소득세는?"
retrive_context = {
    'query' : query 
}

context = retrive(retrive_context)['context']

generate_state = {
    'query' : query,
    'context': context 
}

answer = generate(generate_state)

helpfulness_state = {
    'query': query,
    'answer': answer
}

check_helpfulness_grader(helpfulness_state)

# %%


# %%
graph_builder.add_node('retrieve', retrive)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance, 
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_halluciantion,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)

graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()




