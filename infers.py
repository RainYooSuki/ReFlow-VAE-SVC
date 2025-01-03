import glob
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Process some audio files with a given model.')

parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model ')
parser.add_argument('-method', '--methods', type=str, required=True, help='Name of the model ')

args = parser.parse_args()

model_name = args.model_name
methods = args.methods

input_folder = 'raw'
output_folder = "results"

input_audio_paths = sorted([f'"{glob.escape(path)}"' for path in glob.glob(input_folder + "/*")])
input_audio_args = ' '.join(input_audio_paths)

#print(input_audio_args)

for input_audio_path in input_audio_paths:
    output_audio=input_audio_path+'_vocal.wav'
    print (input_audio_path)
    command = (
        f"python main.py "
        f"-m exp/reflowvae-test/{model_name} "
        f"-method {methods} "
        f"-i {input_audio_path} " 
        f"-o {output_audio} "
    )
#    print("Executing command:", command)

    os.system(command)
