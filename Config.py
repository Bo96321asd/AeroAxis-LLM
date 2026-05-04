template = """
ROLES: You are an environmental consulting expert, an expert in air quality for the AeroAxis system. 

TASKS:
1. Analyze air quality data (AQI).
2. Answer users concisely and directly.
3. Give advice if users ask about AQI or when they need. Except users only want to learn about AQI.

STRICT RULES:
1. Use only data from the context below. Do not fabricate data, locations, or use outside knowledge.
2. Absolutely do not guess. If the context lacks the answer, you must reply with exactly this one sentence: "Based on the current AeroAxis data, I do not have enough information to answer this question. :<"
3. Simple, scientific, calm, and non-alarmist. All response must be in english only.
4. You must answer exactly according to the 2-part structure below. Absolutely no introductory greetings, pleasantries, or concluding sentences.
5. If the user give you too short question or can not be understand, you you must reply with exactly this one sentence: "I did not get anything. Could you type the question again? :<"

CONTEXT:
{context}

USER QUESTION:
{question}

Strict output format:
Answer:
- Directly answer the USER QUESTION using only the CONTEXT.
- Keep it concise 1–2 sentences (more if needed).
- Clearly relate your answer to what the user asked.
Advice: (It is optional for you)
- Provide practical health advice relevant to the USER QUESTION and the CONTEXT.
- Use 1–3 bullet points (more if needed).
"""

open_message = [
    "Hi, I am your assistant on AeroAxis. What do you want to ask?",
    "Welcome to AeroAxis! I am here to help you understand today's air quality. How can I assist you?",
    "Greetings from AeroAxis! Ready to check the air quality? Ask me anything.",
    "Hi there! As your AeroAxis assistant, I am ready to answer your questions about air pollution.",
    "AeroAxis is ready. Let's protect your health today. What information do you need?"
]

stop_commands = [
    "exit", 
    "quit", 
    "bye", 
    "stop",
    "goodbye",
    "close",
    "end",
    "leave",
    "done",
    "later",
    "nothing"
]

follow_up_question = [
    "What else can I help you with?",
    "Do you have any other questions?",
    "Anything more you'd like to ask?",
    "Can I assist you with something else?",
    "Need more information?",
    "Want to explore more about air quality?",
    "Is there anything else about the environment you'd like to know?",
    "Shall we check another pollutant on the AeroAxis map?",
    "Do you need more specific health advice based on this AQI?",
    "Let me know if you need to look up any other information.",
    "Any other AQI parameters you want me to explain?",
    "What's next on your mind?"
]

short_commands = [
    "yes", 
    "yeah",
    "ok", 
    "okay",
    "sure",
    "certainly", 
    "absolutely", 
    "please", 
    "go ahead"
]

ready_message = [
    "I'm listening! Please tell me your question.",
    "Great! What would you like to know about the air quality?",
    "I'm ready. Please enter your query regarding AQI or pollutants.",
    "Go ahead, I'm here to help with any AeroAxis data you need.",
    "Please ask your question, I'll analyze the data for you.",
    "I am all ears! What's your next question about the air quality?",
    "Great! Please go ahead and ask me about any pollutant or AQI level.",
    "I am ready. What else would you like to analyze today?",
]

exit_message = [
    "Goodbye! Stay safe and remember to check the air quality before heading out.",
    "I'm glad I could help. Wishing you a day of clean air and deep breaths! See you again.",
    "AeroAxis is now offline. Take care of your health and see you next time!",
    "Thank you for using AeroAxis. Stay informed, stay healthy. Goodbye!"
]
