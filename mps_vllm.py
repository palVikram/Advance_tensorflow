import multiprocessing as mp
from vllm import LLM, SamplingParams

def generate_text(llm_model, text, output_queue):
    tokenizer = llm_model.get_tokenizer()
    conversations = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False,
    )

    outputs = llm_model.generate(
        [conversations],
        SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")],
        ),
        use_tqdm=False,
    )
    output_queue.put(outputs[0].outputs[0].text)

def process_texts(llm_models, texts):
    mp.set_start_method('spawn')

    output_queue = mp.Queue()
    processes = []

    for text, llm_model in zip(texts, llm_models):
        p = mp.Process(target=generate_text, args=(llm_model, text, output_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not output_queue.empty():
        results.append(output_queue.get())

    return results

if __name__ == '__main__':
    # Initialize LLM models
    llm_models = [
        LLM(
            model="ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16",
            enforce_eager=True,
            gpu_memory_utilization=0.99,
            max_model_len=1024,
        )
        for _ in range(4)
    ]

    # List of texts to be processed
    texts = [
        'Generate a poem about the sun in Spanish',
        'Tell me a story about a brave knight',
        'Explain quantum physics in simple terms',
        'Write a haiku about the ocean'
    ]

    # Process the texts
    results = process_texts(llm_models, texts)

    for result in results:
        print(result)
