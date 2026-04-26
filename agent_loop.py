"""
Agent Loop Implementation for CSE 476 Final Project

The agent loop is designed to use at least 8 inference-time reasoning algorithms
or techniques.
"""


import os, json, textwrap, re, time
import requests
import dotenv

dotenv.load_dotenv()

API_KEY  = os.getenv("OPENAI_API_KEY","")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")  


#copy pasted from given notebook
basesystem="You are a helpful assistant. Reply with only the final answer—no explanation."

def call_model_chat_completions(prompt: str,
                                system= basesystem,
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}


def self_evaluate(question, prediction, expected_answer, model=MODEL):
    """
    Use the model itself as a strict grader.
    Returns True if the model says the prediction matches the expected answer; else False.
    Falls back to a simple normalized string compare if the model's reply is malformed.
    """
    import re

    system = "You are a strict grader. Reply with exactly True or False. No punctuation. No explanation."
    prompt = f"""You are grading a question-answer pair.

Return exactly True if the PREDICTION would be accepted as correct for the EXPECTED_ANSWER.
Otherwise, return False.

QUESTION:
{question}

PREDICTION:
{prediction}

EXPECTED_ANSWER:
{expected_answer}

Answer with exactly: True or False
"""

    r = call_model_chat_completions(
        prompt,
        system=system,
        model=model,
        temperature=0.0,
    )

    reply = (r.get("text") or "").strip().lower()
    if reply.startswith("true"):
        return True
    if reply.startswith("false"):
        return False

    # Fallback: simple normalization-based equality
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return norm(prediction) == norm(expected_answer)




def agent_loop(question: str) -> str:
   
    answer = chain_of_thought(question)
    return f"{answer}"



def chain_of_thought(question: str) -> str:
    #simple chain of thought, just tell the system what you want it to do
    cot_system = "You are an analytical reasoning assistant. Use step-by-step reasoning to solve the question."
    answer=call_model_chat_completions(prompt= question, system= cot_system, model= MODEL,
                                temperature= 0.0,
                                timeout= 60)
    return (answer.get("text") or "").strip()



def best_of_n(question: str, n: int) ->str:
    
    for i in range(n):
        answer=call_model_chat_completions(prompt= question, system= basesystem, model= MODEL,
                                temperature= 0.0,
                                timeout= 60)
        
        validity=self_evaluate(question, answer, expected_answer=question, model=MODEL)
        if validity=="True":
            return answer
        else:
            best_answer=answer
    
    return best_answer

def tree_of_thought(question: str) ->str:
    #use a bfs-like structure to go through steps of a question, returns answer if all steps are true
    
    tot_system="You are an analytical reasoning assistant. " \
    "Use labels STEP for steps and ANSWER for the answer. " \
    "Only output the problem step based off of previous steps, if available. If no more steps needed, give the answer."
    steps=[]
    totqueue=[]
    branch_dict={}
    while totqueue:
        branch=call_model_chat_completions(prompt= question,
                                system= tot_system,
                                model = MODEL,
                                temperature= 0.0,
                                timeout= 60)
        validity=self_evaluate(question, branch, expected_answer=question, model=MODEL)
        if validity=="True":
        
            steps.append(branch)
            branch_dict["all_steps"]=steps
            branch_dict["current_step"]=branch
            branch_dict["validity"]=validity
        
        if "ANSWER" in branch:
            return branch
        else:
            best_answer=branch


    return branch       

def self_consistency(question: str, samples: int = 3) -> str:
    responses = []
    for _ in range(samples):
        result = call_model_chat_completions(
            prompt=question,
            system=basesystem,
            temperature=0.7,
        )
        text = (result.get("text") or "").strip().lower()
        if text:
            responses.append(text)
        time.sleep(0.1)

    if not responses:
        return ""

    return max(set(responses), key=responses.count)

def self_refine(question: str) -> str:
    initial = chain_of_thought(question)
    if not initial:
        return ""

    critique_prompt = (
        f"Question: {question}\n\n"
        f"Proposed answer: {initial}\n\n"
        "Is this answer correct and complete? "
        "Identify any errors or gaps in one or two sentences. "
        "Do not give the corrected answer yet."
    )
    critique_result = call_model_chat_completions(
        prompt=critique_prompt,
        system="You are a careful critic. Be brief and specific.",
        temperature=0.0,
    )
    critique = (critique_result.get("text") or "").strip()

    if not critique:
        return initial

    refine_prompt = (
        f"Question: {question}\n\n"
        f"Initial answer: {initial}\n\n"
        f"Critique: {critique}\n\n"
        "Now give the corrected final answer only. No explanation."
    )
    refined_result = call_model_chat_completions(
        prompt=refine_prompt,
        system="You are a careful solver. Reply only with the final answer, nothing else.",
        temperature=0.0,
    )
    refined = (refined_result.get("text") or "").strip()

    return refined if refined else initial



        

