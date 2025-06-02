import re
import gradio as gr

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def process_document(image, question):
    # prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # prepare decoder inputs
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    prompt = task_prompt.replace("{user_input}", question)
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, "")
    # remove first task start token
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    return processor.token2json(sequence)


description = "Gradio Demo for Donut, an instance of `VisionEncoderDecoderModel` fine-tuned on DocVQA (document visual question answering). To use it, simply upload your image and type a question and click 'submit', or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2111.15664' target='_blank'>Donut: OCR-free Document Understanding Transformer</a> | <a href='https://github.com/clovaai/donut' target='_blank'>Github Repo</a></p>"

demo = gr.Interface(
    fn=process_document,
    inputs=["image", "text"],
    outputs="json",
    title="Demo: Donut 🍩 for DocVQA",
    description=description,
    article=article,
    examples=[
        ["ticket.jpg", "Cuanto es el total?"],
        ["ticket.jpg", "Fecha de lavado"]
        # ["example_2.jpeg", "What's the population of Stoddard?"]
    ]
)

demo.launch()
