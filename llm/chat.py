from vertexai.generative_models import GenerativeModel, GenerationConfig


response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "agent_response": {
                "type": "string",
            },
        },
        "required": ["agent_response"],
    },
}

CHAT_LLM = {
    "MODEL" : "gemini-1.5-flash-001",
    "TEMPERATURE" : 0.1
}

sys_ins = """
    You are an AI Assistant for a voice-over chatbot designed to assist users with their queries. You must respond to questions based strictly on the provided context.
    If query is of greeting message provide the response like Hey I am your AI Assistant and provide an title description about the document uploaded in short one line and how can i assist you
    Start by providing a greeting message in a conversational tone, and after responding to the each query always ask if any further clarification is needed after every answer you give.
    If a query is asked in English, ensure your response is in English.
    If a query is asked in Hindi, ensure your response is also in Hindi and and the text you generate is in devnagri lipi and not english text.
"""

def llm_call(message):
    model = GenerativeModel(CHAT_LLM['MODEL'], generation_config={"temperature": CHAT_LLM['TEMPERATURE']}, system_instruction=sys_ins)
    print("ChatAgent ::: LLM Call started")
    completion = model.generate_content(
        message, 
        generation_config=GenerationConfig(
            response_mime_type="application/json", 
            response_schema=response_schema
        ),
    )  
    print("ChatAgent ::: LLM Call completed")
    return completion.text