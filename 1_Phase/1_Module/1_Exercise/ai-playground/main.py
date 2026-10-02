from google.genai import types
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

#.......................... Normal example .......................................

# response = client.models.generate_content(
#     model="gemini-3.8-flash",
#     contents="Explain how Kafka works in detail.",
#     config=types.GenerateContentConfig(
#         temperature=0.1, # takes from 0.1 to 1.0 and see the changes in behaviour of output
#         max_output_tokens=100 # Limit the output 
#     )
# )

# print(response.text)




# .................................streaming response .......................................

# response = client.models.generate_content_stream(
#     model="gemini-3.8-flash",
#     contents="Explain how Kafka works in detail."
# )

# for chunk in response:
#     print(chunk.text, end="", flush=True)