import openai
from ipywidgets import Text, Button, VBox, Output
from IPython.display import display
import os

# Set your OpenAI API Key
openai.api_key = os.environ.get('OPEN_AI_KEY')

# Function to get a response from the model
def get_completion(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Lower temperature means less randomness
    )
    return response.choices[0].message.content

# Function for handling context-based completions
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

# Initialize conversation context
context = [
    {"role": "system", "content": """
    You are OrderBot, an automated service to collect orders for a pizza restaurant.
    You first greet the customer, then collect the order, and then ask if it's a pickup or delivery.
    You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else.
    If it's a delivery, you ask for an address.
    Finally you collect the payment.
    Make sure to clarify all options, extras, and sizes to uniquely identify the item from the menu.
    You respond in a short, very conversational friendly style.
    The menu includes:
    pepperoni pizza  12.95, 10.00, 7.00
    cheese pizza   10.95, 9.25, 6.50
    eggplant pizza   11.95, 9.75, 6.75
    fries 4.50, 3.50
    greek salad 7.25
    Toppings:
    extra cheese 2.00,
    mushrooms 1.50
    sausage 3.00
    canadian bacon 3.50
    AI sauce 1.50
    peppers 1.00
    Drinks:
    coke 3.00, 2.00, 1.00
    sprite 3.00, 2.00, 1.00
    bottled water 5.00
    """}
]

# Create widgets
input_box = Text(placeholder='Enter your message...')
send_button = Button(description='Send')
output_area = Output()

def on_send_button_clicked(b):
    with output_area:
        user_input = input_box.value.strip()
        if user_input:
            # Display the user message
            print(f"User: {user_input}")
            # Add user message to context
            context.append({"role": "user", "content": user_input})
            # Get the assistant's response
            response = get_completion_from_messages(context)
            context.append({"role": "assistant", "content": response})
            # Display the assistant's message
            print(f"Assistant: {response}\n")
        input_box.value = ''

send_button.on_click(on_send_button_clicked)

# Display the interface
display(VBox([input_box, send_button, output_area]))

# Optionally, initiate the conversation by sending an initial greeting
initial_user_message = "Hi"
context.append({"role": "user", "content": initial_user_message})
initial_response = get_completion_from_messages(context)
context.append({"role": "assistant", "content": initial_response})
with output_area:
    print(f"User: {initial_user_message}")
    print(f"Assistant: {initial_response}\n")
