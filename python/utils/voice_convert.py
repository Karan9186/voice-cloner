import subprocess

def convert_to_wav(input_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ar", "22050",   # sample rate
        "-ac", "1",       # mono
        "-af", "highpass=f=80,lowpass=f=8000",  # remove noise
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)