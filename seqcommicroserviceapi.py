# SEQCOM Engine: Microservice API (FastAPI Wrapper)
# Author: Co-designed with Megan

from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import uvicorn

# Core SEQCOM logic
class SEQCOMEngine:
    def __init__(self):
        self.memory_log = []
        self.reinforcement_scores = {}
        self.graph = nx.DiGraph()

    def log_event(self, domain: str, elapsed_time: float, interventions: List[str], result: Dict, outcome: str):
        signature = f"{domain}:{result['current_phase']}→{result['next_event']}"
        entry = {
            "Domain": domain,
            "ElapsedTime": elapsed_time,
            "Interventions": ", ".join(interventions) if interventions else "None",
            "CurrentPhase": result["current_phase"],
            "NextEvent": result["next_event"],
            "TimeInPhase": round(result["time_in_phase"], 2),
            "PhaseDuration": round(result["phase_duration"], 2),
            "CausalSignature": signature
        }
        self.memory_log.append(entry)

        if signature not in self.reinforcement_scores:
            self.reinforcement_scores[signature] = {"success": 0, "fail": 0}

        self.reinforcement_scores[signature][outcome] += 1
        self.graph.add_edge(f"{domain}:{result['current_phase']}", f"{domain}:{result['next_event']}", label=outcome)

    def compress_memory(self):
        df = pd.DataFrame(self.memory_log)
        return df.groupby("CausalSignature").agg({
            "Domain": "first",
            "CurrentPhase": "first",
            "NextEvent": "first",
            "PhaseDuration": "mean",
            "TimeInPhase": "mean",
            "Interventions": lambda x: list(set(x))
        }).reset_index()

    def suggest_next_action(self, current_phase: str, domain: str):
        df = pd.DataFrame(self.memory_log)
        relevant = df[(df['Domain'] == domain) & (df['CurrentPhase'] == current_phase)]
        if relevant.empty:
            return "No recommendation available."

        grouped = relevant.groupby("NextEvent").size().sort_values(ascending=False)
        best_next = grouped.index[0]
        signature = f"{domain}:{current_phase}→{best_next}"
        scores = self.reinforcement_scores.get(signature, {"success": 0, "fail": 0})
        total = scores['success'] + scores['fail']
        success_rate = scores['success'] / total if total > 0 else 0.0

        return {
            "current_phase": current_phase,
            "predicted_next": best_next,
            "interventions": relevant[relevant["NextEvent"] == best_next]["Interventions"].unique().tolist(),
            "success_rate": round(success_rate, 2),
            "reinforcement": scores
        }

    def predict(self, domain: str, elapsed_time: float, interventions: Optional[List[str]]):
        phases = natural_sequences[domain]
        durations = natural_durations[domain].copy()
        modifiers = intervention_modifiers.get(domain, {})

        if interventions:
            for i in range(len(durations)):
                for intervention in interventions:
                    if intervention in modifiers:
                        durations[i] *= modifiers[intervention]

        cumulative_time = 0
        for phase, duration in zip(phases, durations):
            if elapsed_time <= cumulative_time + duration:
                next_index = phases.index(phase) + 1
                return {
                    "current_phase": phase,
                    "time_in_phase": elapsed_time - cumulative_time,
                    "phase_duration": duration,
                    "next_event": phases[next_index] if next_index < len(phases) else "End"
                }
            cumulative_time += duration
        return {"current_phase": "End", "time_in_phase": 0, "phase_duration": 0, "next_event": None}


# Domain-specific knowledge base
natural_sequences = {
    "banana_ripening": ["Green", "Yellow", "Brown", "Rot"],
    "code_lifecycle": ["Planning", "Development", "Testing", "Deployment", "Decay"],
    "focus_decay": ["Alert", "Engaged", "Distracted", "Fatigued"]
}

natural_durations = {
    "banana_ripening": [2, 2, 2, 0],
    "code_lifecycle": [3, 5, 4, 2, 0],
    "focus_decay": [1, 2, 1, 1]
}

intervention_modifiers = {
    "banana_ripening": {"refrigerate": 1.5, "peel": 0.5, "wrap": 0.8},
    "code_lifecycle": {"hotfix": 0.5, "refactor": 1.2, "skip_testing": 0.6},
    "focus_decay": {"take_break": 1.5, "drink_caffeine": 1.2, "multi_task": 0.7}
}

# FastAPI wrapper
app = FastAPI()
seqcom = SEQCOMEngine()

class PredictionRequest(BaseModel):
    domain: str
    elapsed_time: float
    interventions: Optional[List[str]] = []
    outcome: Optional[str] = "success"

class SuggestionRequest(BaseModel):
    domain: str
    current_phase: str

@app.post("/predict")
def predict(request: PredictionRequest):
    result = seqcom.predict(request.domain, request.elapsed_time, request.interventions)
    seqcom.log_event(request.domain, request.elapsed_time, request.interventions, result, request.outcome)
    return result

@app.post("/suggest")
def suggest(request: SuggestionRequest):
    return seqcom.suggest_next_action(request.current_phase, request.domain)

@app.get("/memory")
def memory():
    return seqcom.compress_memory().to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run("seqcom_api:app", host="127.0.0.1", port=8000, reload=True)

