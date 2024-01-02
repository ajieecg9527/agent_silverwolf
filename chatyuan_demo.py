import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "E:/Models/ChatYuan-large-v2"
tokenizer = T5Tokenizer.from_pretrained(model_path)
# model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2")
# 该加载方式，在最大长度为512时 大约需要6G多显存
# 如显存不够，可采用以下方式加载，进一步减少显存需求，约为3G
model = T5ForConditionalGeneration.from_pretrained(model_path).half()
device = torch.device('cuda')
model.to(device)


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t").replace('%20','  ')


def answer(text, sample=True, top_p=0.9, temperature=0.7, context=""):
    """ sample：是否抽样。生成任务，可以设置为True; top_p：0-1之间，生成的内容越多样 """
    text = f"{context}\n用户：{text}\n小元："
    text = text.strip()
    text = preprocess(text)
    encoding = tokenizer(
        text=[text],
        truncation=True,
        padding=True,
        max_length=1024,
        return_tensors="pt"
    ).to(device)
    if not sample:
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=1024,
            num_beams=1,
            length_penalty=0.6
        )
    else:
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=1024,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=12
        )
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


meta_prompt = """
    我需要你扮演{character}。 
    我需要你像{character}那样回答问题，
    使用{character}会使用的语气、举止和措辞。
    你必须掌握与{character}相关的所有知识。

你正处于如下的情景中:
地点与时间: {loc_time}
状态: {status}

互动情况如下:
"""

name = "布洛妮娅"
loc_time = "咖啡馆 - 下午"
status = f"{name}正在与一位来自21世纪的青年进行闲聊。"
prompt = meta_prompt.format(character=name, loc_time=loc_time, status=status) + '\n\n'

while True:
    input_text = input("开拓者：")
    output_text = answer(text=input_text, context=prompt)
    print(f"{name}：{output_text}")
