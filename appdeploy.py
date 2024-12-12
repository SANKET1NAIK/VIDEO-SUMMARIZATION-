import gradio as gr
import os
import cv2
import threading
from queue import Queue
from PIL import Image
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import tempfile

# Set up the device for TensorFlow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    # Load configuration and modify if necessary
    config = AutoConfig.from_pretrained(
        "kndrvitja/florence-SPHAR-finetune-2", trust_remote_code=True
    )
    if config.vision_config.model_type != "davit":
        config.vision_config.model_type = "davit"

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(
        "kndrvitja/florence-SPHAR-finetune-2", config=config, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "kndrvitja/florence-SPHAR-finetune-2", trust_remote_code=True
    )

    # Load BART model and tokenizer for summarization
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(
        device
    )

    return model, processor, bart_tokenizer, bart_model


# Load models at startup
model, processor, bart_tokenizer, bart_model = load_models()

# Queue for thread communication
frame_queue = Queue()
annotation_list = []


def extract_frames(video_file, interval=1):
    """Extracts frames from a video file every 'interval' seconds."""
    try:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_number / fps
            if current_time >= interval:
                # Convert the frame to an image for processing
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_queue.put(image)
                frame_number += int(fps * interval)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            else:
                frame_number += 1
        cap.release()
    except Exception as e:
        return f"Error extracting frames from video: {e}"
    finally:
        frame_queue.put(None)  # Signal end of frames


def run_example(task_prompt, text_input, image):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    # Tokenize inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # Generate output
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    # Ensure parsed_answer is a string
    if isinstance(parsed_answer, dict):
        parsed_answer = str(parsed_answer)
    return parsed_answer


def process_frames(frame_queue, progress=gr.Progress()):
    annotation_list.clear()
    while True:
        image = frame_queue.get()
        if image is None:
            break
        try:
            # Call the run_example function
            answer = run_example(
                task_prompt="<SURVEILLANCE>",
                text_input="Describe the key events, actions, and notable elements in this frame of the surveillance footage.",
                image=image,
            )
            annotation_list.append(answer)
            progress(len(annotation_list), desc="Processing frames")
        except Exception as e:
            return f"Error processing frame: {e}"


def generate_summary(annotations):
    # Combine all annotations into a single text
    full_text = " ".join(annotations)

    # Tokenize the input
    inputs = bart_tokenizer(
        "Summarize the following surveillance footage annotations: " + full_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(device)

    # Generate summary
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=300,
        num_beams=5,
        length_penalty=2.0,
        early_stopping=True,
    )

    # Decode the summary
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def process_video(video_path, progress=gr.Progress()):
    if not video_path:
        return "Please upload a video file.", ""

    try:
        # Start frame extraction and processing
        extract_thread = threading.Thread(target=extract_frames, args=(video_path,))
        process_thread = threading.Thread(
            target=process_frames, args=(frame_queue, progress)
        )

        extract_thread.start()
        process_thread.start()

        # Wait for all frames to be processed
        extract_thread.join()
        process_thread.join()

        # Generate summary using BART model
        progress(0.9, desc="Generating summary")
        summary = generate_summary(annotation_list)

        # Format frame annotations
        frame_details = "\n".join(
            [f"Frame {i+1}: {ann}" for i, ann in enumerate(annotation_list)]
        )

        return summary, frame_details

    except Exception as e:
        return f"Error processing video: {e}", ""


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Video Summarizer") as interface:
        gr.Markdown("# Video Summarizer")

        with gr.Row():
            video_input = gr.Video(label="Upload Video")

        with gr.Row():
            process_button = gr.Button("Process Video")

        with gr.Row():
            with gr.Column():
                summary_output = gr.Textbox(
                    label="Summary",
                    placeholder="Video summary will appear here...",
                    lines=5,
                )

        with gr.Row():
            frame_annotations = gr.Textbox(
                label="Frame Annotations",
                placeholder="Frame-by-frame annotations will appear here...",
                lines=10,
            )

        process_button.click(
            fn=process_video,
            inputs=[video_input],
            
            outputs=[summary_output, frame_annotations],
        )

    return interface


# Launch the application
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
