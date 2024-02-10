def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Define predefined rules and responses
    greeting_patterns = ["hello", "hi", "hey", "greetings"]
    farewell_patterns = ["bye", "goodbye", "see you", "farewell"]
    inquiry_patterns = ["how are you", "what's up", "how do you do"]

    # Check user input against predefined patterns
    if any(pattern in user_input for pattern in greeting_patterns):
                return "Hello! How can I help you today?"

    elif any(pattern in user_input for pattern in farewell_patterns):
        return "Goodbye! Have a great day!"

    elif any(pattern in user_input for pattern in inquiry_patterns):
        return "I'm just a chatbot, but thanks for asking!"

    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase or ask something else?"

# Example usage:
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    response = simple_chatbot(user_input)
    print("Chatbot:", response)
