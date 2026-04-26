"""
Agent Loop Implementation for CSE 476 Final Project

The agent loop is designed to use at least 8 inference-time reasoning algorithms
or techniques.
"""


import os, json, textwrap, re, time
import requests
import dotenv
import math

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
        "max_tokens": 1024,
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


#modified version of self_evaluate where expected_answer is not needed, llm determines if the answer makes sense in context of question
def self_evaluate(question, prediction, model=MODEL):
    """
    Use the model itself as a strict grader.
    Returns True if the model says the prediction matches the question type; else False.
    Falls back to a simple normalized string compare if the model's reply is malformed.
    """
    import re

    system = "You are a strict grader. Reply with exactly True or False. No punctuation. No explanation."
    prompt = f"""You are grading a question-answer pair.

Return exactly True if the PREDICTION would be accepted as correct for the QUESTION based on question type.
Otherwise, return False.

QUESTION:
{question}

PREDICTION:
{prediction}

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
    return norm(prediction) 



def agent_loop(question: str) -> str:
   
    answer = route_question(question)
    return f"{answer}"



def chain_of_thought(question: str) -> str:
    #simple chain of thought, just tell the system what you want it to do
    cot_system = "You are an analytical reasoning assistant. Use step-by-step reasoning to solve the question. Output with answer only. Do not put explanation in the answer."
    answer=call_model_chat_completions(prompt= question, system= cot_system, model= MODEL,
                                temperature= 0.5,
                                timeout= 60)
    return (answer.get("text") or "").strip()

def best_of_n(question: str, n: int) ->str:
    
    for i in range(n):
        answer=call_model_chat_completions(prompt= question, system= basesystem, model= MODEL,
                                temperature= 0.5,
                                timeout= 60)
        
        validity=self_evaluate(question=question, prediction=answer["text"], model=MODEL)
        if validity:
            return answer["text"]
        else:
            best_answer=answer
    
    return best_answer["text"]

def tree_of_thought(question: str) ->str:
    #use a bfs-like structure to go through steps of a question, returns answer if all steps are true
    
    tot_system=f"""You are a reasoning assistant that breaks down problems into steps.
      Use labels STEP for steps and ANSWER for the answer. 
      Only output the current step of the given problem based off of previous steps, if available. Label each step as "STEP: "
      If no more steps needed, give the answer and label as "ANSWER: "."""
    steps=[]
    totqueue=[]

    #adds first step as root node
    #we assume the first step is automatically valid so we can reduce llm calls
    branch=call_model_chat_completions(prompt= question,
                                system= tot_system,
                                model = MODEL,
                                temperature= 0.5,
                                timeout= 60)
    #limit number of llm calls to 15 per question to account for the few shot model in routing function
    totalcalls=1
    steps.append(branch["text"])
    #root has its own dict, and the leafs have their own
    root_dict={}
    root_dict["all_steps"]=steps
    root_dict["current_step"]=branch["text"]
    root_dict["validity"]=True
    if branch["text"] != "None":
        totqueue.append(root_dict)
    
    best_answer=branch["text"]
   
    
    while totqueue and totalcalls <=15:
        item=totqueue.pop()
        newprompt=question+"Steps: "+ "\n".join(item["all_steps"])
        branch=call_model_chat_completions(prompt= newprompt,
                                system= tot_system,
                                model = MODEL,
                                temperature= 0.5,
                                timeout= 60)
        totalcalls+=1
        validity=self_evaluate(question=question, prediction=branch["text"], model=MODEL)
        totalcalls+=1
        if validity and branch["text"] != "None": 
            branch_dict={}       
            # steps.append(branch)
            branch_dict["all_steps"]=item["all_steps"]
            branch_dict["current_step"]=branch["text"]
            branch_dict["validity"]=validity
            if "ANSWER" in branch["text"]:
                #outputs everything after answer label
                return branch["text"].split("ANSWER: ")[-1]
            totqueue.append(branch_dict)      
        else:
            best_answer=branch["text"]


    return best_answer      

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

def return_final_math_answer(question: str) ->str:
    math_final_answer_prompt = "You are a data extraction bot. You must read the following input and extract the final mathemtatical answer. Your response should be ONLY the final result found in the text, either a number or variable. Do not include any other explanation. Here is the input:\n\n"

    resp = call_model_chat_completions(
        prompt=question, 
        system=math_final_answer_prompt 
    )
    return resp.get("text", question).strip()

def extract_final_answer(text: str) -> str:
    final_answer_prompt = "You are a data extraction bot. You must read the following input and extract the final answer. Your response should be ONLY the final result found in the text, either a word, variable, or concise phrase. Do not include any other explanation. Here is the input:\n\n"

    resp = call_model_chat_completions(
        prompt=text, 
        system=final_answer_prompt 
    )
    return resp.get("text", text).strip()

def calculator(exp: str) -> str:
    """Basic math evaluator."""
    try:
        final_expression = exp.replace('$', '').replace(',', '').strip()
        return str(eval(final_expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error w calculator: {e}"

def tool_augmented_reasoning(question: str) -> str:
    math_assistant_prompt = (
        "You are a math assistant. Think step-by-step to solve the problem. "
        "Whenever you need to do arithmetic, put the expression inside double square brackets ike this: [[328 / 2]]. I will provide the result. "
        "Once you have the final answer, state it clearly."
    )
    
    current_prompt = question
    
    for i in range(6):
        resp = call_model_chat_completions(prompt=current_prompt, system=math_assistant_prompt)
        text = resp.get("text", "")
        calculation_required = re.search(r"\[\[(.*?)\]\]", text)
        
        if calculation_required:
            equation = calculation_required.group(1)
            result = calculator(equation)
            current_prompt += f"\nAssistant: {text}\nResult of [[{equation}]]: {result}"
        else:
            return return_final_math_answer(text)
            
    return return_final_math_answer(text)

        

def least_to_most(question: str) -> str:
    prompt = "Question: " + question + "\n\n" + "Break down the following problem into smaller sub-problems that need to be solved in order to reach a final answer. List the sub-problems you have identified."
    resp = call_model_chat_completions(
        prompt=prompt, 
        system="You are a logical reasoning assistant. Reduce the problem into simpler sub-problems."
    )
    steps = resp.get("text", "")

    new_prompt = ("Question: " + question + "\n\n" + "Sub-problems:\n" + steps + "\n\n" + "Now solve each sub-problem step by step to reach the final answer. Provide only the final answer in your response.")

    final_response = call_model_chat_completions(
        prompt=new_prompt, 
        system="You are a logical reasoning assistant. Solve the sub-problems step by step to reach the final answer. Reply only with the final answer."
    )

    return extract_final_answer(final_response.get("text", ""))

def route_question(question: str) -> str:
    category = few_shot_prompt_classifier(question)
    if category == "TOOL_AUGMENTED":
        return tool_augmented_reasoning(question)
    elif category == "TREE_OF_THOUGHT": 
        return tree_of_thought(question)
    elif category == "BEST_OF_N":
        return best_of_n(question, n=5)
    elif category == "SELF_REFINE":
        return self_refine(question)
    elif category == "SELF_CONSISTENCY":
        return self_consistency(question, samples=5)
    elif category == "LEAST_TO_MOST":
        return least_to_most(question)
    elif category == "CHAIN_OF_THOUGHT":
        return chain_of_thought(question)
    else:
        return chain_of_thought(question) 


def few_shot_prompt_classifier(question: str) -> str:
      prompt = f"""You are a question routing assistant."
      "Classify the following question into one of these categories: 
       "1. TOOL_AUGMENTED: Multi-step math or problems requiring precise arithmetic.\n"
        "2. SELF_REFINE: Coding tasks, debugging, or technical implementations.\n"
        "3. TREE_OF_THOUGHT: Complex planning, logic puzzles, or branching scenarios.\n"
        "4. CHAIN_OF_THOUGHT: General knowledge, common sense questions, explanations, or simple reasoning.\n\n"
        "5. BEST_OF_N: Multiple choice questions, ambiguous queries, or when multiple valid answers exist.\n\n"
        "6. SELF_CONSISTENCY: When the question asks you to make a future prediction. When the question is open-ended, subjective, or likely to have multiple valid perspectives.\n\n"
        "7. LEAST_TO_MOST: For complex multi-step problems or sequential tasks that can be decomposed into a simpler set of subproblems."
        "EXAMPLES:\n"
        "Q: How many even integers between 4000 and 7000 have four different digits?\n"
        "A: TOOL_AUGMENTED\n\n"
        "Q: Scramble the letters in each word of a given text, keeping the first and last letters of each word intact.\nNote that: Notes: Words are determined by regex word boundaries. The scrambling only affects words longer than three characters, leaving shorter words unchanged.\nThe function should output with:\n    str: The scrambled text.\nYou should write self-contained code starting with:\n```\nimport random\nimport re\ndef task_func(text, seed=None):\n```\n"
        "A: SELF_REFINE\n\n"
        "Q: You are an agent that can predict future events. The event to be predicted: \"Will Mathieu van der Poel win the Green Jersey at the 2025 Tour de France? (around 2025-07-28T07:59:00Z). Will Mathieu van der Poel win the Green Jersey at the 2025 Tour de France?'\n"
        "A: SELF_CONSISTENCY\n\n"
        "Q: In what show did Cynthia Nixon receive the 2004 Primetime Emmy Award for Outstanding Supporting Actress in a Comedy Series and a Screen Actors Guild Award for her performance?\n"
        "A: CHAIN_OF_THOUGHT\n\n"
        "Q: Which of the following options is a common household pet? A. Car B. Dog. C. Apple D. Mosquito\n"
        "A: BEST_OF_N\n\n"
        "For long winded or complex questions, recommmend TREE_OF_THOUGHT.""
        "For sequential multi-step problems that do not involve math, but require reasoning, recommend LEAST_TO_MOST."
        "Now classify the following question. Reply with ONLY the category name."""
        
      resp = call_model_chat_completions(prompt=question, system=prompt)
 
      return (resp.get("text") or "").strip().upper()

if __name__ == "__main__":

    test_question = "Do people with swallowing disorders need high viscosity drinks? Facts: Swallowing disorders can make thin liquids like water dangerous to drink. Liquid thickeners are marketed towards people with difficulty drinking."
    
    print(f"question:\n{test_question}\n")
    
    final_answer = few_shot_prompt_classifier(test_question)
    
    print("answer: \n")
    print(final_answer)