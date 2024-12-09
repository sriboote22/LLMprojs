from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub


# Set up Hugging Face LLM (you can also load models locally if required)
# Replace "model_id" with any model available on Hugging Face Hub (e.g., "gpt2" or "distilgpt2")
llm = HuggingFaceHub(huggingfacehub_api_token="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",repo_id="facebook/m2m100_418M")

# Create a prompt template
template = "Translate the sentence '{sentence}' into {target_language}."
prompt = PromptTemplate(
    input_variables=["sentence", "target_language"],
    template=template
)

# Create an LLMChain with the Hugging Face model
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
output = chain.run({
    "sentence": "Hello, how are you?",
    "target_language": "French"
})

# Print the output
print(output)
