import argparse
import os
import yt_dlp
import whisper
import time
import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Setup the argument parser
parser = argparse.ArgumentParser(description="Download audio from YouTube, transcribe it using Whisper, summarize the transcription with OpenAI's GPT, and clean up the downloaded files.")
parser.add_argument("video_url", help="The URL of the YouTube video")
parser.add_argument("video_name", help="It will be used as name of output files")
parser.add_argument("whisper_model_size", default="base", help="The size of the Whisper model to use")
parser.add_argument("gpt_model", default="gpt-4", help="The GPT model to use for summarization")
args = parser.parse_args()

# Load the .env file
load_dotenv()

# Initialize OpenAI API key
client = OpenAI()
client.api_key = os.getenv('OPENAI_API_KEY')

def format_timestamp(seconds):
    return datetime.datetime.utcfromtimestamp(seconds).strftime('%H:%M:%S,%f')[:-3]

def split_transcription(transcription, chunk_size=50):
    chunks = []
    entries = transcription.strip().split('\n\n')
    for i in range(0, len(entries), chunk_size):
        chunk = '\n\n'.join(entries[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def generate_translation(chunk, gpt_model, language):
    messages = [
        {"role": "system", "content": f"Translate the following subtitles to {language}. Keep the format intact."},
        {"role": "user", "content": chunk}
    ]
    response = client.chat.completions.create(model=gpt_model, messages=messages)
    translation = response.choices[0].message.content
    return translation.strip()

def merge_translations(translations):
    return '\n\n'.join(translations)

# Download audio from YouTube using yt-dlp
def download_audio(video_url):
    # Define the output template for the original file
    original_file_template = 'original_downloaded_file.%(ext)s'

    options = {
        'format': 'bestaudio/best',  # Choose the best quality audio format
        'extractaudio': True,  # Only keep the audio
        'audioformat': 'wav',  # Convert to wav
        'outtmpl': original_file_template,  # Use the defined template
        'noplaylist': True,  # Only download single video and not a playlist
        'postprocessors': [{  # Postprocessors for extracting and converting audio
            'key': 'FFmpegExtractAudio',  # Extract audio using FFmpeg
            'preferredcodec': 'wav',  # Specify the desired codec
            'preferredquality': '192',  # Specify the quality
        }, {
            'key': 'FFmpegMetadata',  # Add metadata to the file (optional)
        }]
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video_url])

    # Define the names for the final files
    input_file = "original_downloaded_file.wav"
    output_file = "output.wav"

    # Convert the audio to mono, 16kHz, and 16-bit using FFmpeg directly
    ffmpeg_command = f"ffmpeg -i {input_file} -ac 1 -ar 16000 -sample_fmt s16 {output_file}"
    print("ffmpeg command: {}".format(ffmpeg_command))
    os.system(ffmpeg_command)

    if os.path.exists(input_file):
        os.remove(input_file)

    return output_file

# Transcribe audio using Whisper
def transcribe_audio(audio_path, model_size):
    model = whisper.load_model(model_size)
    model = model.cuda()
    result = model.transcribe(audio_path)

    srt_entries = []
    for i, segment in enumerate(result["segments"], start = 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"]
        srt_entries.append(f"{i}\n{start} --> {end}\n{text}\n")

    out = "\n".join(srt_entries)

    with open("trans.txt", 'w', encoding="utf-8") as file:
        file.write(out)
    return out

# Generate a summary using OpenAI GPT
def generate_translation(transcription, gpt_model, language):
    # Construct the initial conversation context
    messages = [
        {"role": "system", "content": "You are a helpful assistant tasked with translating the subtitles. DON'T CHANGE FORMAT OF FILE. Always translate full file, don't stop in middle. The correct name is ZYLIA - if you notice any similar work, fix it."},
        {"role": "user", "content": f"Please translate this transcription to {language} language: {transcription}"}
    ]

    attempts = 0
    max_attempts = 3
    response = None

    while attempts < max_attempts:
        try:
            # Make the API call to generate the completion
            response = client.chat.completions.create(
                model=gpt_model,
                messages=messages
            )
            # Break the loop if the response is successful
            break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            time.sleep(1)  # Wait a bit before retrying to avoid overwhelming the server

    if response:
        # Extract the assistant's reply from the response
        summary = response.choices[0].message.content
        return summary.strip()
    else:
        # Return a default message or raise an error if all attempts fail
        return "Failed to generate a summary after several attempts."

# Clean up the downloaded audio file
def clean(audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Deleted the file: {audio_path}")
    else:
        print(f"The file {audio_path} does not exist")

# Save the summary to a text file
def save_translation(summary, filename):
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(summary)
    print(f"Translation saved to {filename}")

# Main process
if __name__ == "__main__":
    trans_path = Path('./trans.txt')
    transcription = ""

    if trans_path.exists():
        transcription = trans_path.read_text()
        print("Transcription file exists, reading instead of downloading.")
    else:
        audio_path = download_audio(args.video_url)
        transcription = transcribe_audio(audio_path, args.whisper_model_size)

    transcription_chunks = split_transcription(transcription)

    for lang in {"korean", "japanese", "chinese"}:
        translations = []
        for chunk in transcription_chunks:
            translation_chunk = generate_translation(chunk, args.gpt_model, lang)
            translations.append(translation_chunk)
        
        final_translation = merge_translations(translations)
        
        # Save final translation to a text file
        translation_filename = f"{args.video_name}_{lang}.txt"
        with open(translation_filename, 'w', encoding="utf-8") as file:
            file.write(final_translation)
        print(f"Translation saved to {translation_filename}")

    # Clean up the downloaded audio if it was downloaded
    if 'audio_path' in locals():
        clean(audio_path)