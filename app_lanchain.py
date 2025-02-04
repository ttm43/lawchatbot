import os
import json
from flask import Flask, render_template, request, jsonify, session
from mike_langchain import kb_df, real_estate_chatbot, extract_variable_from_response, format_variable, is_user_feedback, refine_answer_with_gpt, LLMWrapper, run_evaluation
from langchain_openai import ChatOpenAI

app = Flask(__name__, template_folder='template')
app.secret_key = '159753test'  # set session key


# llm = ChatOpenAI(
#     temperature=0,
#     model_name="gpt-4o-mini",
#     openai_api_key='xxxxxx'  # Replace with your actual API key
# )
# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    model_name="glm-4-flash",
    openai_api_key='xxxxxxx',  # Replace with your actual API key
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
llm_wrapper = LLMWrapper(llm)

@app.route('/')
def home():
    session['current_state'] = 'initial'
    session['best_match_index'] = None
    session['required_variables'] = []
    return render_template('chat.html')

@app.route('/greeting', methods=['GET'])
def greeting():
    welcome_message = ("¡Bienvenido! Soy un asistente virtual especializado en leyes inmobiliarias españolas. "
                       "¿En qué puedo ayudarte hoy? Por favor, hazme cualquier pregunta sobre temas legales "
                       "relacionados con bienes raíces en España.")
    return jsonify({"response": welcome_message})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    print(f"Received user message: {user_message}")
    
    current_state = session.get('current_state', 'initial')
    best_match_index = session.get('best_match_index')
    required_variables = session.get('required_variables', [])
    
    if current_state == "waiting_for_variable":
        if not required_variables:
            return jsonify({"response": "Lo siento, hubo un error en el procesamiento. ¿Puede reformular su pregunta original?"})

        variable = required_variables[0]
        variable_type = variable['variable_type']
        print(f"Variable type: {variable_type}")
        variable_value = extract_variable_from_response(user_message, variable_type, llm_wrapper)
        print(f"Extracted variable value: {variable_value}")
        if not variable_value:
            return jsonify({"response": f"Lo siento, no pude entender su respuesta. ¿Podría proporcionar los detalles correctos de {variable_type}?"})

        formatted_variable = format_variable(variable_type, variable_value)
        print(f"Formatted variable: {formatted_variable}")
        matched_row = kb_df.iloc[best_match_index]
        print(f"Matched row: {matched_row}")
        variable_exists = False
        if variable_type in ["PROVINCIA", "COMUNIDAD_AUTONOMA"]:
            for answers in matched_row['answers']:
                if answers[variable_type.lower()] == formatted_variable:
                    variable_exists = True
        elif variable_type == "DATE":
            for answers in matched_row['answers']:
                time_info = answers['time'].get("time", {})
                if 'ini' in time_info and 'end' in time_info:
                    if time_info['ini'] <= formatted_variable <= time_info['end']:
                        variable_exists = True

        if variable_exists:
            initial_answer = matched_row['answers']
            final_answer = refine_answer_with_gpt(llm_wrapper, user_message, initial_answer)
            session['current_state'] = 'initial'
            session['best_match_index'] = None
            session['required_variables'] = []
            return jsonify({"response": f"{final_answer}\n¿Era la respuesta que estabas buscando? ¿O tienes más preguntas?"})
        else:
            session['current_state'] = 'initial'
            session['best_match_index'] = None
            session['required_variables'] = []
            return jsonify({"response": "Lo siento, no tenemos una respuesta específica para esta variable. Estoy consultando con un asistente legal para obtener más información. ¿Tiene alguna otra pregunta mientras tanto?"})

    input_data = json.dumps({
        'user_input': user_message,
        'current_state': current_state,
        'best_match_index': best_match_index,
        'required_variables': required_variables
    })

    response = real_estate_chatbot(input_data, llm_wrapper)

    if isinstance(response, dict):
        output = response.get('output', '')
        session['current_state'] = response.get('current_state', 'waiting_for_question')
        session['best_match_index'] = response.get('best_match_index')
        session['required_variables'] = response.get('required_variables', [])
    else:
        output = response
        session['current_state'] = 'waiting_for_question'

    return jsonify({"response": output})

if __name__ == '__main__':
    app.run(debug=True)