import os

directory = '/mnt/yiran/LongRoPE/script/longrope_scale'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            filedata = file.read()

        new_data = filedata.replace(' ', ',')

        with open(filepath, 'w') as file:
            file.write(new_data)