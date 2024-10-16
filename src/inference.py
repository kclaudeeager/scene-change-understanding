import torch
from utils.visualization import plot_bbox

TASK_PROMPTS = {
    'CAPTION': '<CAPTION>',
    'DETAILED_CAPTION': '<DETAILED_CAPTION>',
    'MORE_DETAILED_CAPTION': '<MORE_DETAILED_CAPTION>',
    'OD': '<OD>',
    'DENSE_REGION_CAPTION': '<DENSE_REGION_CAPTION>',
    'REGION_PROPOSAL': '<REGION_PROPOSAL>',
    'CAPTION_TO_PHRASE_GROUNDING': '<CAPTION_TO_PHRASE_GROUNDING>'
}

def run_inference(model, processor, task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def analyze_image(model, processor, image, task='CAPTION', text_input=None, visualize=False):
    task_prompt = TASK_PROMPTS.get(task, TASK_PROMPTS['CAPTION'])
    result = run_inference(model, processor, task_prompt, image, text_input)
    
    if visualize and task in ['OD', 'DENSE_REGION_CAPTION']:
        plot_bbox(image, result[task_prompt])
    
    return result

def caption_image(model, processor, image):
    return analyze_image(model, processor, image, task='CAPTION')

def detailed_caption_image(model, processor, image):
    return analyze_image(model, processor, image, task='DETAILED_CAPTION')

def more_detailed_caption_image(model, processor, image):
    return analyze_image(model, processor, image, task='MORE_DETAILED_CAPTION')

def object_detection(model, processor, image, visualize=False):
    return analyze_image(model, processor, image, task='OD', visualize=visualize)

def dense_region_caption(model, processor, image, visualize=False):
    return analyze_image(model, processor, image, task='DENSE_REGION_CAPTION', visualize=visualize)

def region_proposal(model, processor, image):
    return analyze_image(model, processor, image, task='REGION_PROPOSAL')

def caption_to_phrase_grounding(model, processor, image, caption):
    return analyze_image(model, processor, image, task='CAPTION_TO_PHRASE_GROUNDING', text_input=caption)