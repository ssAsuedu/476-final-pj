# 476-final-pj

To run this project, create a .env file that contains your API key with the following name:

OPENAI_API_KEY=<YOUR_API_KEY>

To run the script, in the root directory start a python virtual environment and download the required libraries. Here are the commands for mac:

python3 -m venv .venv                               
source .venv/bin/activate

Download requirements:

pip install requests python-dotenv jupyter ipykernel

Then, run the following command to execute the script:

python generate_answer_template.py  

When the lab has run successfully, you will see the following message in the terminal:

Wrote 20 answers to cse_476_final_project_answers.json and validated format successfully.

Check the answers JSON file to see the results of the LLM's performance.