from vllm import LLM, SamplingParams

llm1 = LLM(
    model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
    enforce_eager=True,
    gpu_memory_utilization=0.99,
    max_model_len=1024,
)
llm2 = LLM(
    model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
    enforce_eager=True,
    gpu_memory_utilization=0.99,
    max_model_len=1024,
)
llm3 = LLM(
    model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
    enforce_eager=True,
    gpu_memory_utilization=0.99,
    max_model_len=1024,
)
llm4 = LLM(
    model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
    enforce_eager=True,
    gpu_memory_utilization=0.99,
    max_model_len=1024,
)

def llm1_func(text):
    tokenizer = llm1.get_tokenizer()
    conversations = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False,
    )
    outputs = llm1.generate(
        [conversations],
        SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")],
        ),
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text

def llm2_func(text):
    tokenizer = llm2.get_tokenizer()
    conversations = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False,
    )
    outputs = llm2.generate(
        [conversations],
        SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")],
        ),
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text

def llm3_func(text):
    tokenizer = llm3.get_tokenizer()
    conversations = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False,
    )
    outputs = llm3.generate(
        [conversations],
        SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")],
        ),
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text

def llm4_func(text):
    tokenizer = llm4.get_tokenizer()
    conversations = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False,
    )
    outputs = llm4.generate(
        [conversations],
        SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")],
        ),
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text


import multiprocessing as mp

def process_texts(texts):
    # Define the list of functions
    functions = [llm1_func, llm2_func, llm3_func, llm4_func]
    
    # Ensure the number of texts and functions match
    assert len(texts) == len(functions), "Number of texts must match the number of functions"

    # Initialize a pool of workers
    with mp.Pool(processes=len(functions)) as pool:
        # Map the texts to the functions
        results = pool.starmap(lambda f, text: f(text), zip(functions, texts))

    return results

if __name__ == '__main__':
    # List of texts to be processed
    texts = [
        'Generate a poem about the sun in Spanish',
        'Tell me a story about a brave knight',
        'Explain quantum physics in simple terms',
        'Write a haiku about the ocean'
    ]

    # Process the texts
    results = process_texts(texts)

    for result in results:
        print(result)

