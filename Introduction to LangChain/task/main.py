import dotenv
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load API keys as environment variables
dotenv.load_dotenv()

# Initialize Groq chat LLM API
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2
)
# Load a Prompt Template in the form a Q & A
example_template = PromptTemplate.from_template("Q: {input}\nA: {output}")

# Provide example for the few-shot Prompt
examples = [
    {"input": "Sun", "output": "The Sun is the star at the center of our solar system."},
    {"input": "Moon",
     "output": "The Moon is a planetary-mass object or satellite planet. It is the earth's only natural satellite."},
]

# Initialize Few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Provide key details of the input. Provide relevant facts each line by line. Maximum 3 lines.",
    suffix="Q: {question}\nA:",
    input_variables=["question"],
)

# Accept user input
user_input = input().lower()

# Generate String Prompt formatting it with input
final_prompt = few_shot_prompt.format(question=user_input)

# Invoke LLM and receive output.
response = llm.invoke(final_prompt)

# Print output content
print(response.content)
