import os
import pandas as pd
import json
import chromadb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
from functools import lru_cache
from typing import List, Any, Dict
from langchain.llms import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.smith import RunEvalConfig, run_on_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "xxxxxxx"  # Replace with your actual API key

# Database and ChromaDB setup
DATABASE_URL = "postgresql://user:159753@localhost/mike"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
client = chromadb.HttpClient("http://localhost:8000")
collection = client.get_or_create_collection("mike-prod")

# Read knowledge base data
query = "SELECT id, name, question_samples, required_variables_data, answers, tag_ids, required_variables FROM new_question_knowledge;"
kb_df = pd.read_sql(query, engine)

class LLMWrapper:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate a response using the LLM"""
        messages = [HumanMessage(content=prompt)]
        response = self.llm.generate([messages])
        logger.info(f"Token usage: {response.llm_output['token_usage']}")
        return response.generations[0][0].text
# Initialize LLM (using OpenAI as an example, can be easily replaced with other LLMs)
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="glm-4-flash",
    openai_api_key='xxxxxxxxxx',  # Replace with your actual API key
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
llm_wrapper = LLMWrapper(llm)

@lru_cache(maxsize=100)
def cached_matching_question_from_chromadb(text: str) -> List[str]:
    """Query matching text chunks from ChromaDB"""
    results = collection.query(query_texts=[text], n_results=2)
    return [item for sublist in results['documents'] for item in sublist]

def matching_rows_index_from_kb_df(matched_text_chunks: List[str]) -> List[int]:
    """Find matching row indices from the knowledge base DataFrame"""
    kb_df['question_sample_texts'] = kb_df['question_samples'].apply(
        lambda x: [sample.get('sample', '') for sample in x] if isinstance(x, list) else []
    )
    kb_exploded_df = kb_df.explode('question_sample_texts')
    matching_indices = kb_exploded_df[kb_exploded_df['question_sample_texts'].isin(matched_text_chunks)].index.unique().tolist()
    logger.info(f'matching_indices: {matching_indices}')
    return matching_indices

def is_related_to_real_estate(text: str, llm_wrapper: LLMWrapper) -> bool:
    """Determine if the text is related to Spanish real estate law"""
    prompt = f"Determine if the following text is related to Spanish real estate law:\n\n'{text}'\n\nPlease answer only 'yes' or 'no'."
    response = llm_wrapper.generate(prompt, max_tokens=5)
    return "yes" in response.lower()

def refine_answer_with_gpt(llm_wrapper: LLMWrapper, question: str, raw_answer: str) -> str:
    """Refine the answer using the LLM"""
    prompt = f"Please refine the following legal answer by using the question: {question} to make it more formal and user-friendly:\n\n{raw_answer}, only response the answer in spanish"
    return llm_wrapper.generate(prompt, max_tokens=100)

def find_best_match_index(indices: List[int], llm_wrapper: LLMWrapper, text: str) -> int:
    """Find the best matching index"""
    names = [kb_df.loc[index, 'name'] for index in indices]
    names_string = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
    prompt = f"""
    Please compare the similarity between the given query: "{text}" and the following classifications:

    {names_string}

    Analyze the semantic similarity, considering synonyms and related concepts. Then, rank these classifications from most similar (1) to least similar ({len(names)}).

    Respond ONLY with the rankings in the format:
    1. [index of most similar]
    2. [index of second most similar]
    ...
    {len(names)}. [index of least similar]

    Do not include any explanations or additional text in your response.
    """
    rankings_text = llm_wrapper.generate(prompt, max_tokens=10)
    rankings = [int(line.split('. ')[1]) - 1 for line in rankings_text.split('\n')]
    return indices[rankings[0]]

def extract_variable_from_response(user_input: str, variable_type: str, llm_wrapper: LLMWrapper) -> str:
    """Extract variable from user input"""
    prompt = f"Extract the {variable_type} from the following response: \n\n'{user_input}', only response the value, not the type"
    return llm_wrapper.generate(prompt, max_tokens=10).strip()

def format_variable(variable_type, variable_value):
    if variable_type in ["PROVINCIA", "COMUNIDAD_AUTONOMA"]:
        return variable_value.upper().replace(" ", "_")
    elif variable_type == "DATE":
        # Assuming input date in "YYYY-MM-DD" format, convert to ISO 8601 format
        date_obj = datetime.datetime.strptime(variable_value, "%Y-%m-%d")
        return date_obj.isoformat() + "Z"
    return variable_value

def is_user_feedback(user_input: str, llm_wrapper: LLMWrapper) -> bool:
    """Determine if the user input is feedback"""
    prompt = f"""
    Determine if the following user input is feedback to a previous answer or a new question about Spanish real estate law.
    
    User input: "{user_input}"
    
    If the input is feedback (such as expressing gratitude, understanding, or satisfaction), respond with "True".
    If the input is a new question or request for information, respond with "False".
    
    Respond with only one word: either "True" or "False".
    """
    result = llm_wrapper.generate(prompt, max_tokens=10).strip().upper()
    return result == "TRUE"

def real_estate_chatbot(input_data: str, llm_wrapper: LLMWrapper) -> Dict[str, Any]:
    """Main chatbot function"""
    data = json.loads(input_data)
    user_input = data['user_input']
    current_state = data.get('current_state', 'initial')
    best_match_index = data.get('best_match_index')
    required_variables = data.get('required_variables', [])

    logger.info(f"Processing user input: {user_input}")
    logger.info(f"Current state: {current_state}")
    logger.info(f"Best match index: {best_match_index}")
    logger.info(f"Required variables: {required_variables}")

    if is_user_feedback(user_input, llm_wrapper):
        logger.info("User input identified as feedback")
        return {
            "output": "Me alegro de que la respuesta haya sido útil. ¿Tiene alguna otra pregunta sobre leyes inmobiliarias españolas?",
            "current_state": "waiting_for_question",
            "best_match_index": None,
            "required_variables": []
        }

    if current_state in ['initial', 'waiting_for_question']:
        if not is_related_to_real_estate(user_input, llm_wrapper):
            return {
                "output": "Este problema no está relacionado con las leyes inmobiliarias españolas. Por favor, haga una pregunta sobre leyes inmobiliarias españolas.",
                "current_state": "waiting_for_question",
            }
        matched_text_chunks = cached_matching_question_from_chromadb(user_input)
        logger.info(f"Matched text chunks: {matched_text_chunks}")
        if not matched_text_chunks:
            return {
                "output": "No se encontró información relacionada con su pregunta. Por favor, reformule o proporcione más detalles.",
                "current_state": "waiting_for_question",
            }

        matching_index = matching_rows_index_from_kb_df(matched_text_chunks)
        logger.info(f"Matching indices: {matching_index}")
        if not matching_index:
            return {
                "output": "No pude encontrar una respuesta adecuada en nuestra base de conocimientos. ¿Podría reformular su pregunta?",
                "current_state": "waiting_for_question",
            }

        best_match_index = find_best_match_index(matching_index, llm_wrapper, user_input)
        logger.info(f"Best match index: {best_match_index}")
        if best_match_index is None:
            return {
                "output": "No pude encontrar una respuesta adecuada. ¿Podría reformular su pregunta?",
                "current_state": "waiting_for_question",
            }

        best_match = kb_df.iloc[best_match_index]
        required_variables = best_match.get('required_variables_data', [])
        logger.info(f"Required variables: {required_variables}")
        if not required_variables:
            initial_answer = best_match['answers']
            final_answer = refine_answer_with_gpt(llm_wrapper, user_input, initial_answer)
            return {
                "output": f"{final_answer}\n¿Era la respuesta que estabas buscando? ¿O tienes más preguntas?",
                "current_state": "answered",
                "best_match_index": best_match_index,
            }
        else:
            return {
                "output": f"Para responder a su pregunta, necesito algunos detalles adicionales. ¿Podría proporcionar información sobre {required_variables[0]['variable_type']}?",
                "current_state": "waiting_for_variable",
                "best_match_index": best_match_index,
                "required_variables": required_variables,
            }

    return {
        "output": "Lo siento, no pude procesar su pregunta correctamente. ¿Podría reformularla?",
        "current_state": "waiting_for_question",
    }

# Add evaluation configuration
eval_config = RunEvalConfig(
    evaluators=[
        "qa",
        "context_qa",
        RunEvalConfig.Criteria("relevance"),
        RunEvalConfig.Criteria("coherence"),
    ]
)

# Function to run evaluation
def run_evaluation(dataset_name: str):
    run_on_dataset(dataset_name=dataset_name, llm=llm, eval_config=eval_config)