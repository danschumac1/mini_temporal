from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# save the base model

# Set the working directory
#os.chdir('/home/dan/mini_temporal')
#model_path = '/home/dan/mini_temporal/training/models/AQA/gemma-2b/relevant_context'
model_path = '/home/dan/mini_temporal/training/models/AQA/gemma-2b/no_context'
os.path.exists(model_path)

# Ensure padding side is set to 'right' to avoid potential overflow issues
tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=True, add_eos_token=False)
tokenizer.padding_side = 'left'

# Specify the new paths
# model_path = './training/models/gemma-2b/no_context'

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_path)

# Load the adapter model
adapter_model = PeftModel.from_pretrained(base_model, model_path)

# Example text generation function
def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=16,
        do_sample=True, 
        top_k=50, 
        temperature=.8, 
        num_beams=4,
        repetition_penalty=2.5, 
        length_penalty=.1
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return generated_text

# Test the function with a prompt
#prompt = """<start_of_turn>user\nContext: In 2004, Anthony Rios was named president of the Disney-ABC Television Group. She became one of the most powerful women in the entertainment industry, overseeing a vast portfolio that included the ABC Television Network, Disney Channel Worldwide, ABC Family, and SOAPnet, among others. Rios's tenure at Disney-ABC was marked by her focus on expanding the company's digital presence and embracing new technologies to distribute content.\nQuestion: Who was named president of Disney-ABC television group in 2004?<end_of_turn>\n<start_of_turn>model\nThe correct answer is"""
#prompt =  """<start_of_turn>user\nQUESTION": "A car bomb explodes the hotel in which city in Indonesia in August 2003? Here is the context: A car bomb exploded at the JW Marriott Hotel in Jakarta, Indonesia, on August 5, 2003. This terrorist attack killed 12 people and injured 150 others. The attack was attributed to the Islamist militant group Jemaah Islamiyah, which had links to al-Qaeda.<end_of_turn>\n<start_of_turn>model\nThe answer is"""
#prompt = """Who is the president of the united states? The answer is: """
#prompt = """Answer the following question using the context\nQuestion:\nLater in January 1993, which American politician succeeded George H W Bush and became the forty-second President of the USA?\nHere is the context: Later in January 1993, Bill Clinton succeeded George H. W. Bush and became the forty-second President of the USA.\nAnswer: """
#prompt = """Later in January 1993, which American politician succeeded George H W Bush and became the forty-second President of the USA? Here is the context: Later in January 1993, Bill Clinton succeeded George H. W. Bush and became the forty-second President of the USA. The answer is: """
prompt = """Answer the following question using the context.\nQuestion: The United States was the largest donor of food aid to what country during the drought of 1984-85?\nHere is the context: During the drought of 1984-85, Niger faced a severe food crisis, with millions of people suffering from famine and malnutrition. The United States emerged as the largest donor of food aid to Niger, providing significant support to alleviate the humanitarian crisis. The drought had devastating effects on the country's agricultural production, leading to widespread food shortages and hunger. International aid efforts, including contributions from the United States, played a crucial role in addressing the urgent food needs of the affected population in Niger.\n The answer is:"""
 
# prompt = """Who was named president of Disney-ABC television group in 2004?"""

generated_text = generate_text(prompt, adapter_model, tokenizer)#.split('model\n')
print(generated_text)
