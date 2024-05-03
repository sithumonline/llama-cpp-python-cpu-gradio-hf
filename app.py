import os
import gradio as gr
import copy
from llama_cpp import Llama
from huggingface_hub import hf_hub_download  


llm = Llama(
    model_path=hf_hub_download(
        repo_id=os.environ.get("REPO_ID", "microsoft/Phi-3-mini-4k-instruct-gguf"),
        filename=os.environ.get("MODEL_FILE", "Phi-3-mini-4k-instruct-q4.gguf"),
    ),
    n_ctx=2048,
    n_gpu_layers=50, # change n_gpu_layers if you have more or less VRAM 
) 

# history = []

# system_message = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """


# def generate_text(message, history):
def generate_text(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    temp = ""
    input_prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n "
    for interaction in history:
        input_prompt = input_prompt + str(interaction[0]) + " [/INST] " + str(interaction[1]) + " </s><s> [INST] "

    input_prompt = input_prompt + str(message) + " [/INST] "

    output = llm(
        input_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=max_tokens,
        stop=[
            "<|prompter|>",
            "<|endoftext|>",
            "<|endoftext|> \n",
            "ASSISTANT:",
            "USER:",
            "SYSTEM:",
        ],
        stream=True,
    )
    for out in output:
        stream = copy.deepcopy(out)
        temp += stream["choices"][0]["text"]
        yield temp

    # history = ["init", input_prompt]


demo = gr.ChatInterface(
    generate_text,
    title="llama-cpp-python on GPU",
    description="Running LLM with https://github.com/abetlen/llama-cpp-python",
    examples=[
        ['How to setup a human base on Mars? Give short answer.'],
        ['Explain theory of relativity to me like I’m 8 years old.'],
        ['What is 9,000 * 9,000?'],
        ['Write a pun-filled happy birthday message to my friend Alex.'],
        ['Justify why a penguin might make a good king of the jungle.']
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

#demo.queue(concurrency_count=1, max_size=5)?

if __name__ == "__main__":
    demo.launch()

