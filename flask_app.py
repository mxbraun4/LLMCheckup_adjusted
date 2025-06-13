"""The app main."""
import uuid
from datetime import datetime
from os.path import isfile, join

import gin
import json
import logging
import os
import traceback
import random
import torch
import gc
import threading
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
else:
    # For gunicorn/production, set spawn method if not already set
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, which is fine
        pass

from flask import Flask, request, Blueprint, render_template, jsonify, make_response
from logging.config import dictConfig

from actions.prediction.predict import convert_str_to_options
from actions.util_functions import text2int, get_current_prompt
from logic.core import ExplainBot
from logic.sample_prompts_by_action import sample_prompt_for_action
from timeout import timeout, TimeoutError

import easyocr
import numpy as np
from scipy.io.wavfile import read
import librosa
import soundfile as sf

import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

from PIL import Image

my_uuid = uuid.uuid4()


# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

bp = Blueprint('host', __name__, template_folder='templates')

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch optimizations for better performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable optimized cuDNN operations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow asynchronous CUDA operations

# Memory management optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.enabled = True    # Enable cuDNN acceleration

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot
BOT = ExplainBot()

# Setup logger
logger = logging.getLogger(__name__)



@bp.route('/')
def home():
    """Load the explanation interface."""
    app.logger.info("Loaded Login")
    objective = BOT.conversation.describe.get_dataset_objective()

    BOT.conversation.build_temp_dataset()

    df = BOT.conversation.temp_dataset.contents['X']
    f_names = list(BOT.conversation.temp_dataset.contents['X'].columns)

    dataset = BOT.conversation.describe.get_dataset_name()

    entries = []
    for j in range(10):
        temp = {}
        for f in f_names:
            if dataset == "ECQA" and f == "choices":
                temp[f] = convert_str_to_options(df[f][j])
            else:
                temp[f] = df[f][j]
        entries.append(temp)

    return render_template("index.html", currentUserId="user", datasetObjective=objective, entries=entries,
                           dataset=dataset)


@bp.route("/log_feedback", methods=['POST'])
def log_feedback():
    """Logs feedback"""
    feedback = request.data.decode("utf-8")
    app.logger.info(feedback)
    split_feedback = feedback.split(" || ")

    message = f"Feedback formatted improperly. Got: {split_feedback}"
    assert split_feedback[0].startswith("MessageID: "), message
    assert split_feedback[1].startswith("Feedback: "), message
    assert split_feedback[2].startswith("Username: "), message
    assert split_feedback[3].startswith("Answer: "), message

    message_id = split_feedback[0][len("MessageID: "):]
    feedback_text = split_feedback[1][len("Feedback: "):]
    username = split_feedback[2][len("Username: "):]
    answer = split_feedback[3][len("Answer: "):]

    current_time = datetime.now()
    time_stamp = current_time.timestamp()
    date_time = datetime.fromtimestamp(time_stamp)
    str_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

    logging_info = {
        "id": message_id,
        "feedback_text": feedback_text,
        "username": username,
        "answer": answer,
        "dataset": BOT.conversation.describe.get_dataset_name(),
        "parsed_text": BOT.parsed_text,
        "user_text": BOT.user_text,
        "timestamp": str_time
    }

    BOT.log(logging_info)
    return ""


@bp.route("/export_history", methods=["POST"])
def export_history():
    """Export chat history as JSON file."""
    try:
        # Get the chat history
        history = BOT.get_chat_history()
        
        # Create a JSON response
        response = make_response(json.dumps(history, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=chat_history.json'
        return response
    except Exception as e:
        logger.error(f"Error exporting history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route("/sample_prompt", methods=["POST"])
def sample_prompt():
    """Samples a prompt"""
    data = json.loads(request.data)
    action = data["action"]
    username = data["thisUserName"]

    prompt = sample_prompt_for_action(action,
                                      BOT.prompts.filename_to_prompt_id,
                                      BOT.prompts.final_prompt_set,
                                      BOT.conversation)
    logging_info = {
        "username": username,
        "requested_action_generation": action,
        "generated_prompt": prompt
    }
    BOT.log(logging_info)

    return prompt


@bp.route("/cancel_generation", methods=['POST'])
def cancel_generation():
    """Cancel the current model generation."""
    # Since we're no longer using multiprocessing, just clean up CUDA resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return json.dumps({"status": "cancelled"})


@bp.route("/get_response", methods=['POST'])
def get_bot_response():
    """Load the box response."""
    if request.method == "POST":
        response = ""
        try:
            flag = None
            audio = None

            try:
                # Receive the uploaded image
                img = request.files["image"]
                flag = "img"
            except:
                pass
            try:
                data = json.loads(request.data)
                flag = "text"
            except:
                pass

            try:
                audio = request.files["audio"]
                audio.save("recording.wav")
                flag = "audio"
            except:
                pass

            if flag == "img":
                # Save image locally
                img.save(f"./{img.filename}")
                app.logger.info(f"Image uploaded!")

                if torch.cuda.is_available():
                    gpu = True
                else:
                    gpu = False
                reader = easyocr.Reader(['en'], gpu=gpu)  # this needs to run only once to load the model into memory

                result = reader.readtext(f"{img.filename}", detail=0)

                if len(result) < 2:
                    raise ValueError("Only one sentence is recognized. Please try other images!")
                else:
                    temp = {'first_input': result[0], 'second_input': result[1]}

                    BOT.conversation.custom_input = temp
                    BOT.conversation.used = False
                    app.logger.info(f"[CUSTOM INPUT] {temp}")
                    response = "You have given a custom input via uploaded image. " \
                               "Please enter a follow-up question or prompt! <br><br>" + "Entered custom input: <br>"
                    if BOT.conversation.describe.get_dataset_name() == "covid_fact":
                        response += f"Claim: {temp['first_input']} <br>Evidence: {temp['second_input']} <>"
                    else:
                        response += f"Question: {temp['first_input']} <br>Choices: {temp['second_input']} <>"
            elif flag == "audio":
                model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
                processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

                # Load wav in array
                x, _ = librosa.load('./recording.wav', sr=16000)
                sf.write('tmp.wav', x, 16000)

                a = read("tmp.wav")
                temp = np.array(a[1], dtype=np.float64)
                inputs = processor(temp, sampling_rate=16000, return_tensors="pt")
                generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Convert words to digits if available
                user_text = text2int(transcription[0])

                BOT.user_text = user_text
                conversation = BOT.conversation

                # Remove generated wav files
                os.remove("./recording.wav")
                os.remove("./tmp.wav")

                app.logger.info("generating the bot response")
                response = f"<b>Recorded text</b>: {user_text}<br><br>"

            elif flag == "text":
                # Change level for QA
                level = data["qalevel"]
                prompt_type = data["prompt_type"]

                BOT.conversation.qa_level = level
                BOT.conversation.prompt_type = prompt_type
                app.logger.info(f"Prompt type: {prompt_type}")

                # Normal user input
                if data['custom_input'] == '0':
                    # Normal user input
                    user_text = data["userInput"]
                    BOT.user_text = user_text
                    conversation = BOT.conversation

                    app.logger.info("generating the bot response")
                    # Generate response directly (timeout removed to avoid multiprocessing issues)
                    response = BOT.update_state(user_text, conversation)
                elif data['custom_input'] == '1':
                    # custom input
                    user_text = data["userInput"]

                    if BOT.conversation.describe.get_dataset_name() == "ECQA":
                        if len(user_text["second_input"].split("-")) != 5:
                            return "5 choices should be provided and concatenated by '-'!"

                    BOT.conversation.custom_input = user_text
                    BOT.conversation.used = False
                    app.logger.info(f"[CUSTOM INPUT] {user_text}")
                    response = "You have given a custom input. " \
                               "Please enter a follow-up question or prompt! <br><br>" + "Entered custom input: <br>"
                    if BOT.conversation.describe.get_dataset_name() == "covid_fact":
                        response += f"Claim: {user_text['first_input']} <br>Evidence: {user_text['second_input']} <>"
                    else:
                        response += f"Question: {user_text['first_input']} <br>Choices: {user_text['second_input']} <>"

                    BOT.conversation.store_last_parse(f"custominput '{user_text}'")
                else:
                    # custom input removal
                    app.logger.info(f"[CUSTOM INPUT] Custom input is removed!")
                    BOT.conversation.custom_input = None
                    BOT.conversation.used = True
                    response = "Entered custom input is now removed! <>"
                BOT.write_to_history(BOT.user_text, response)
        except torch.cuda.OutOfMemoryError:
            # Clean up CUDA resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            response = "I recognized a CUDA out of memory. I suggest to choose a smaller " \
                       "model for your hardware configuration or with GPTQ/4bit quantization. You can do that by " \
                       "opening the global_config.gin file and editing the value of GlobalArgs.config to an " \
                       "equivalent with a model of smaller parameter " \
                       "size, e.g. \"ecqa_llama_gptq.gin\" or \"ecqa_pythia.gin\"."
        except Exception as ext:
            # Clean up CUDA resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            app.logger.error(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.error(f"Exception getting bot response: {ext}")
            
            # Check if it's a CUDA-related error
            if "CUDA" in str(ext) or "cuda" in str(ext):
                response = "I encountered a CUDA processing error. Please try restarting the application. If the problem persists, ensure your CUDA drivers are properly installed."
            else:
                response = random.choice(BOT.dialogue_flow_map["sorry"])
        return response


@bp.route("/custom_input", methods=["POST"])
def custom_input():
    try:
        data = request.get_json()
        claim = data.get('claim')
        evidence = data.get('evidence')
        
        if not claim or not evidence:
            return jsonify({'success': False, 'error': 'Missing claim or evidence'})
        
        # Process the custom input
        response = BOT.process_custom_input(claim, evidence)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logger.error(f"Error processing custom input: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route("/filter_dataset", methods=["POST"])
def filter_dataset():
    filter_text = json.loads(request.data)["filterMsgText"]
    df = BOT.conversation.stored_vars["dataset"].contents["X"]
    if len(filter_text) > 0:
        filtered_df = df[df[BOT.text_fields].apply(lambda row: row.str.contains(filter_text)).any(axis=1)]

        BOT.conversation.temp_dataset.contents["X"] = filtered_df
        app.logger.info(f"{len(filtered_df)} instances of {BOT.conversation.describe.dataset_name} include the filter "
                        f"string '{filter_text}'")
        final_df = filtered_df
    else:
        final_df = df
    return {
        'jsonData': final_df.to_json(orient="index"),
        'totalDataLen': len(df)
    }


@bp.route("/reset_temp_dataset", methods=["POST"])
def reset_temp_dataset():
    """Reset the temporary dataset."""
    try:
        BOT.conversation.build_temp_dataset()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error resetting dataset: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route("/get_prompt", methods=["POST"])
def get_prompt():
    """Get the current prompt."""
    try:
        prompt = get_current_prompt(BOT.parsed_text, BOT.conversation)
        return jsonify({'success': True, 'prompt': prompt})
    except Exception as e:
        logger.error(f"Error getting prompt: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/update_knowledge_level', methods=['POST'])
def update_knowledge_level():
    try:
        data = request.get_json()
        level = data.get('level')
        if level not in ['beginner', 'expertise', 'expert']:
            return jsonify({'success': False, 'error': 'Invalid knowledge level'})
        
        # Update the model's knowledge level setting
        # Store the knowledge level in conversation for tutorial operations
        BOT.conversation.qa_level = level
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating knowledge level: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/get_current_prompt')
def get_current_prompt_route():
    try:
        prompt = get_current_prompt(BOT.parsed_text, BOT.conversation)
        return jsonify({'success': True, 'prompt': prompt})
    except Exception as e:
        logger.error(f"Error getting current prompt: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/update_prompt_type', methods=['POST'])
def update_prompt_type():
    try:
        data = request.get_json()
        prompt_type = data.get('type')
        if prompt_type not in ['none', 'zero-shot', 'plan-solve', 'opro']:
            return jsonify({'success': False, 'error': 'Invalid prompt type'})
        
        # Update the prompt type
        # Store the prompt type in conversation
        BOT.conversation.prompt_type = prompt_type
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating prompt type: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/update_additional_prompt', methods=['POST'])
def update_additional_prompt():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        # Update the additional prompt
        # Store the additional prompt in conversation
        BOT.conversation.prompt_type = prompt
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating additional prompt: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle file uploads (images and audio)."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Process the file based on its type
        if file.content_type.startswith('image/'):
            # Handle image file
            response = process_image_file(file)
        elif file.content_type.startswith('audio/'):
            # Handle audio file
            response = process_audio_file(file)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type'})
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/process_audio', methods=['POST'])
def process_audio():
    """Process audio data from recording."""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio data provided'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No audio data selected'})
        
        # Process the audio file
        response = process_audio_file(audio_file)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def process_image_file(file):
    """Process an uploaded image file."""
    try:
        # Read the image
        image = Image.open(file)
        
        # Process the image (e.g., OCR for text)
        text = extract_text_from_image(image)
        
        return {
            'success': True,
            'text': text,
            'type': 'image'
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def process_audio_file(file):
    """Process an uploaded audio file."""
    try:
        # Read the audio file
        audio_data, sample_rate = sf.read(file)
        
        # Process the audio (e.g., speech-to-text)
        text = transcribe_audio(audio_data, sample_rate)
        
        return {
            'success': True,
            'text': text,
            'type': 'audio'
        }
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def extract_text_from_image(image):
    """Extract text from an image using OCR."""
    try:
        # Initialize OCR reader
        reader = easyocr.Reader(['en'])
        
        # Perform OCR
        results = reader.readtext(np.array(image))
        
        # Extract text from results
        text = ' '.join([result[1] for result in results])
        
        return text
    except Exception as e:
        logger.error(f"Error in OCR: {str(e)}")
        raise

def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio using speech-to-text model."""
    try:
        # Initialize speech-to-text model
        processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        
        # Process audio
        inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
        
        # Generate transcription
        generated_ids = model.generate(inputs["input_features"])
        
        # Decode transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcription
    except Exception as e:
        logger.error(f"Error in speech-to-text: {str(e)}")
        raise

app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    app.run(debug=False, port=4000, host="0.0.0.0")
