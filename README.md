## Peripheral Memory for LLMs [ICML 2025]

### How to run
1. Download the original LLMs and store them locally: [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b), [Llama3 (8B)](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Gemma (2B)](https://huggingface.co/google/gemma-2-2b-it), [Phi3 (3.8B)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
2. Update the corresponding model path in `CONFIG`
3. Perform editing by executing `python memory.py --llm_name gpt-j --data_name zsre`, (You can specify the `llm_name` and `data_name` according to your requirements)

### Citation
```
@inproceedings{zhaimengchenwangqiicml2025,
    author = {Songlin Zhai and Yuan Meng and Yongrui Chen and Yiwei Wang and Guilin Qi},
    title = {Peripheral Memory for LLMs: Integration of Sequential Memory Banks with Adaptive Querying},
    booktitle = {ICML},
    year = {2025}
}
```
