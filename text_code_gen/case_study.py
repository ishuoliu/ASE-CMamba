from transformers import GPT2Tokenizer


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("../huggingface/microsoft/CodeGPT-small-java")

    gold_file = "./saved_models/CodeGPT/java/small/lr5e-5/test.gold"

    codegpt_file = "./saved_models/CodeGPT/java/small/lr5e-5/test.output"
    codegpt_adapted_file = "./saved_models/CodeGPT/java/small-adapted/lr5e-5/test.output"
    mamba_file = "../saved_models/text_code_gen/mamba/java/130m/lr2e-4/test.output"
    mamba_2_file = "../saved_models/text_code_gen/mamba2/java/130m/lr1e-4/test.output"

    with open(gold_file, 'r') as f0:
        gold_content = f0.readlines()
    with open(codegpt_file, 'r') as f1:
        codegpt_content = f1.readlines()
    with open(codegpt_adapted_file, 'r') as f2:
        codegpt_adapted_content = f2.readlines()
    with open(mamba_file, 'r') as f3:
        mamba_content = f3.readlines()
    with open(mamba_2_file, 'r') as f4:
        mamba_2_content = f4.readlines()

    total_len = 0
    for item in mamba_2_content:
        total_len += len(tokenizer.encode(item))

    print(total_len / 2000)

