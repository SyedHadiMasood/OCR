import gradio as gr
import os
import uuid
import shutil
from transformers import AutoModel, AutoTokenizer

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'srimanth-d/GOT_CPU', 
    trust_remote_code=True, 
    low_cpu_mem_usage=True, 
    use_safetensors=True, 
    pad_token_id=tokenizer.eos_token_id
)
model.eval()

# Define upload folder
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to handle OCR process
def run_GOT(image, got_mode):
    unique_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
    shutil.copy(image, image_path)

    try:
        # Perform OCR based on selected mode
        if got_mode == "plain texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='ocr')
        elif got_mode == "format texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='format')
        
        return res
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# Gradio app
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>OCR by Hadi</h1>")

    # Upload option and dropdown
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload your image")
        task_dropdown = gr.Dropdown(
            choices=["plain texts OCR", "format texts OCR"],
            label="Select OCR Mode",
            value="plain texts OCR"
        )
        submit_button = gr.Button("Submit")
    
    # Output text area for OCR result
    ocr_result = gr.Textbox(label="OCR Output")

    # Handle submit button click
    submit_button.click(
        run_GOT,
        inputs=[image_input, task_dropdown],
        outputs=[ocr_result]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()