import subprocess
import os

# Define the environment token
os.environ['HF_TOKEN'] = ''  # need to create the token in Hugging Face

# Define the command to run the script
command = ['python', r"C:\Users\emily\OneDrive\Documents\GitHub\CVLface\cvlface\apps\verification\verify.py", '--data_root', r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)

# Show the command output
print(result.stdout)
print(result.stderr)
