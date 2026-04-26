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

def self_evaluate_tests(tests, model=MODEL, grader_model=None, sleep_sec=0.2, verbose=True):
    """
    Run the tests by querying the model for each prompt, then use LLM-as-a-judge
    (self_evaluate) to determine correctness.

    Args:
        tests: list of dicts with keys: id, prompt, expected (and optionally type)
        model: model used to generate predictions
        grader_model: model used to judge correctness (defaults to `model` if None)
        sleep_sec: small delay between calls to be polite to the API
        verbose: if True, print a summary line per test

    Returns:
        rows: list of dicts with fields:
              id, expected, got, correct, status, error
    """
    import time

    judge_model = grader_model or model
    rows = []

    for t in tests:
        # 1) Get model prediction
        r = call_model_chat_completions(
            t["prompt"],
            system="You are a careful solver. Reply ONLY with the final answer, nothing else.",
            model=model,
            temperature=0.0,
        )
        got = (r.get("text") or "").strip()

        # 2) LLM-as-a-judge: strict True/False
        is_correct = self_evaluate(
            question=t["prompt"],
            prediction=got,
            expected_answer=t["expected"],
            model=judge_model,
        )

        row = {
            "id": t.get("id", "<unnamed>"),
            "expected": t["expected"],
            "got": got,
            "correct": bool(is_correct),
            "status": r.get("status"),
            "error": r.get("error"),
        }
        rows.append(row)

        if verbose:
            mark = "✅" if is_correct else "❌"
            print(f"{mark} {row['id']}: expected={row['expected']!r}, got={row['got']!r} (HTTP {row['status']})")
            if row["error"]:
                print("   error:", row["error"])

        if sleep_sec:
            time.sleep(sleep_sec)

    return rows


def agent_loop(question: str) -> str:
   
    answer = best_of_n(question, n=3)
    return f"{answer}"



def chain_of_thought(question: str) -> str:
    #simple chain of thought, just tell the system what you want it to do
    cot_system = "You are an analytical reasoning assistant. Use step-by-step reasoning to solve the question."
    answer=call_model_chat_completions(prompt= question, system= cot_system, model= MODEL,
                                temperature= 0.5,
                                timeout= 60)
    return answer["text"]


def best_of_n(question: str, n: int) ->str:
    
    for i in range(n):
        answer=call_model_chat_completions(prompt= question, system= basesystem, model= MODEL,
                                temperature= 0.5,
                                timeout= 60)
        
        validity=self_evaluate(question, answer, expected_answer=question, model=MODEL)
        if validity:
            return answer["text"]
        else:
            best_answer=answer
    
    return best_answer["text"]

def tree_of_thought(question: str) ->str:
    #use a bfs-like structure to go through steps of a question, returns answer if all steps are true
    
    tot_system="You are an analytical reasoning assistant. " \
    "Use labels STEP for steps and ANSWER for the answer. " \
    "Only output the problem step based off of previous steps, if available. If no more steps needed, give the answer."
    steps=[]
    totqueue=[]

    #adds first step as root node
    #we assume the first step is automatically valid so we can reduce llm calls
    branch=call_model_chat_completions(prompt= question,
                                system= tot_system,
                                model = MODEL,
                                temperature= 0.5,
                                timeout= 60)
    steps.append(branch)
    #root has its own dict, and the leafs have their own
    root_dict={}
    root_dict["all_steps"]=steps
    root_dict["current_step"]=branch
    root_dict["validity"]=True
    totqueue.append(root_dict)
   
    
    while totqueue:
        item=totqueue.pop()
        newprompt=question+"Steps: "+ "\n".join(item["all_steps"])
        branch=call_model_chat_completions(prompt= newprompt,
                                system= tot_system,
                                model = MODEL,
                                temperature= 0.5,
                                timeout= 60)
        validity=self_evaluate(question, branch, expected_answer=question, model=MODEL)
        if validity: 
            branch_dict={}       
            # steps.append(branch)
            branch_dict["all_steps"]=item["all_steps"]
            branch_dict["current_step"]=branch
            branch_dict["validity"]=validity
            if "ANSWER" in branch:
                return branch
            totqueue.append(branch_dict)      
        else:
            best_answer=branch


    return best_answer      



def tool_augmented_reasoning(question: str) ->str:

    return ""

MATH_KEYWORDS = [
    "calculate", "compute", "evaluate", "solve", "how many", "probability", "difference", "$","¥", "£", "€", "+", "-", "*", "/", "equation", "formula", "=", "find the", "ration", "average", "product"
]
PLANNING_KEYWORDS = [
    "[plan]", "[statement]", "actions"
] 
CODING_KEYWORDS = [
    "code", " def ", "task_func", "implement", "algorithm", "function", "class", "write self-contained"
]   
LOGIC_KEYWORDS = [
    "exchange", "complete the rest of", "swap"
]
CONTEXT_KEYWORDS = [
   "facts:", "context:", "[doc]"
]
COMMON_SENSE_KEYWORDS = [
    "can", "could", "would", "should", "were", "does", "did"
]
FUTURE_PREDICTION_KEYWORDS = [
    "predict", "will happen", "\\boxed\{your_prediction\}", "predict future events"
]


def is_mcq(question: str) -> bool:
    question = question.lower()
    return any(option in question for option in ["a: ", " (a) ", "options:" , "a. ", "a)"])


def classify_question(question: str) -> str:
    question = question.lower()
    mcq = is_mcq(question)
   
    if  any(keyword in question for keyword in PLANNING_KEYWORDS):
        return "tree_of_thought"
    
    if any(keyword in question for keyword in CODING_KEYWORDS):
        return "self_refine"
    
    if is_mcq(question) or any(keyword in question for keyword in COMMON_SENSE_KEYWORDS):
        return "best_of_n"
    
    if any(c in question for c in MATH_KEYWORDS) or any(char.isdigit() for char in question):
        return "tool_augmented_reasoning"

    return "chain_of_thought"

def route_question(question: str) -> str:
    question = question.lower()
    route = classify_question(question)
    if route == "chain_of_thought":
        return chain_of_thought(question)
    elif route == "tree_of_thought":
        return tree_of_thought(question)
    elif route == "best_of_n":
        return best_of_n(question, n=5)
    elif route == "tool_augmented_reasoning":
        return tool_augmented_reasoning(question)
        

