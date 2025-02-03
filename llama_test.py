from llama_cpp import Llama

llm = Llama(
    # model_path="./data/models/DeepseekQwen/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    model_path='./data/models/llama3_1-8B/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf', # replace this with the Llama3.2 3B's path'
    n_gpu_layers=-1,  # Uncomment to use GPU acceleration
    seed=1337,  # Uncomment to set a specific seed
    n_ctx=4096,  # Uncomment to increase the context window
)
output = llm(
    'Q: What is the meaning of life in 3 lines? A: ',  # Prompt
    max_tokens=150,  # set to None to generate up to the end of the context window
    stop=['Q:', '\n'],  # Stop generating just before the model would generate a new question
    echo=True,  # Echo the prompt back in the output
)  # Generate a completion, can also call create_completion
print(output)
