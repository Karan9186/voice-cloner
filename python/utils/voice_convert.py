import subprocess

def convert_to_wav(input_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ac", "1",          # mono
        "-ar", "16000",      # sample rate
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)