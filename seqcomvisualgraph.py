# SEQCOM Engine: Real-Time Visualization of Causal Trace
# Author: Co-designed with Megan

from typing import List, Dict, Optional, Generator
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class SEQCOMEngine:
    def __init__(self):
        self.memory_log = []
        self.reinforcement_scores = {}  # Map: CausalSignature -> {'success': int, 'fail': int}
        self.graph = nx.DiGraph()

    def log_event(self, domain: str, elapsed_time: float, interventions: Optional[List[str]], result: Dict, outcome: Optional[str] = "success"):
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

        if outcome == "success":
            self.reinforcement_scores[signature]["success"] += 1
        else:
            self.reinforcement_scores[signature]["fail"] += 1

        # Update real-time causal graph
        self.graph.add_edge(f"{domain}:{result['current_phase']}", f"{domain}:{result['next_event']}", label=f"{outcome}")

    def compress_memory(self):
        df = pd.DataFrame(self.memory_log)
        compressed = df.groupby("CausalSignature").agg({
            "Domain": "first",
            "CurrentPhase": "first",
            "NextEvent": "first",
            "PhaseDuration": "mean",
            "TimeInPhase": "mean",
            "Interventions": lambda x: list(set(x))
        }).reset_index()
        return compressed

    def suggest_next_action(self, current_phase: str, domain: str) -> Optional[str]:
        df = pd.DataFrame(self.memory_log)
        relevant = df[(df['Domain'] == domain) & (df['CurrentPhase'] == current_phase)]
        if relevant.empty:
            return "No recommendation available. Try logging more events."

        grouped = relevant.groupby("NextEvent").size().sort_values(ascending=False)
        best_next = grouped.index[0]
        interventions = relevant[relevant["NextEvent"] == best_next]["Interventions"].unique()

        signature = f"{domain}:{current_phase}→{best_next}"
        scores = self.reinforcement_scores.get(signature, {"success": 0, "fail": 0})
        total = scores["success"] + scores["fail"]
        success_rate = scores["success"] / total if total > 0 else 0.0

        return (
            f"From phase '{current_phase}', most likely next is '{best_next}'.\n"
            f"Past interventions: {interventions.tolist()}\n"
            f"Reinforcement score: success={scores['success']}, fail={scores['fail']}, success rate={success_rate:.2f}"
        )

    def ingest_stream(self, event_stream: Generator[Dict, None, None]):
        for event in event_stream:
            result = seqcom_predict(event['domain'], event['elapsed_time'], event['interventions'])
            self.log_event(event['domain'], event['elapsed_time'], event['interventions'], result, outcome=event.get('outcome', 'success'))
            print(f"Ingested: {event['domain']}@{result['current_phase']} → {result['next_event']}")

    def visualize_graph(self):
        plt.figure(figsize=(12, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw(self.graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        plt.title("SEQCOM Real-Time Causal Trace Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# Sequence prediction setup
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


def seqcom_predict(domain: str, elapsed_time: float, interventions: Optional[List[str]] = None):
    phases = natural_sequences[domain]
    durations = natural_durations[domain].copy()
    modifiers = intervention_modifiers.get(domain, {})

    if interventions:
        for i, duration in enumerate(durations):
            for intervention in interventions:
                if intervention in modifiers:
                    durations[i] *= modifiers[intervention]

    cumulative_time = 0
    for phase, adjusted_duration in zip(phases, durations):
        if elapsed_time <= cumulative_time + adjusted_duration:
            return {
                "current_phase": phase,
                "time_in_phase": elapsed_time - cumulative_time,
                "phase_duration": adjusted_duration,
                "next_event": phases[phases.index(phase) + 1] if phases.index(phase) + 1 < len(phases) else "End"
            }
        cumulative_time += adjusted_duration

    return {"current_phase": "End", "time_in_phase": 0, "phase_duration": 0, "next_event": None}


# Example: Real-time event stream ingestion + visualization
if __name__ == '__main__':
    engine = SEQCOMEngine()

    def event_stream():
        yield {"domain": "banana_ripening", "elapsed_time": 2.5, "interventions": ["peel"], "outcome": "success"}
        yield {"domain": "code_lifecycle", "elapsed_time": 6, "interventions": ["hotfix"], "outcome": "fail"}
        yield {"domain": "focus_decay", "elapsed_time": 2.2, "interventions": ["drink_caffeine", "multi_task"], "outcome": "success"}

    engine.ingest_stream(event_stream())

    print("\nCompressed Memory:")
    print(engine.compress_memory())

    print("\nPlanning Suggestion:")
    print(engine.suggest_next_action("Testing", "code_lifecycle"))

    print("\nVisualizing SEQCOM Causal Graph...")
    engine.visualize_graph()
