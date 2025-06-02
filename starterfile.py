# SEQCOM Engine: Sequential Compression Memory System (Testable Version with Planning Reinforcer)
# Author: Co-designed with Megan

from typing import List, Dict, Optional
import pandas as pd

# STEP 1: Define a SEQCOM Engine with Reinforcer
class SEQCOMEngine:
    def __init__(self):
        self.memory_log = []

    def log_event(self, domain: str, elapsed_time: float, interventions: Optional[List[str]], result: Dict):
        entry = {
            "Domain": domain,
            "ElapsedTime": elapsed_time,
            "Interventions": ", ".join(interventions) if interventions else "None",
            "CurrentPhase": result["current_phase"],
            "NextEvent": result["next_event"],
            "TimeInPhase": round(result["time_in_phase"], 2),
            "PhaseDuration": round(result["phase_duration"], 2),
            "CausalSignature": f"{domain}:{result['current_phase']}→{result['next_event']}"
        }
        self.memory_log.append(entry)

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
        """
        Recommend next best action based on reinforced memory signatures.
        """
        df = pd.DataFrame(self.memory_log)
        relevant = df[(df['Domain'] == domain) & (df['CurrentPhase'] == current_phase)]
        if relevant.empty:
            return "No recommendation available. Try logging more events."
        most_common = relevant["NextEvent"].value_counts().idxmax()
        interventions = relevant[relevant["NextEvent"] == most_common]["Interventions"].unique()
        return f"From phase '{current_phase}', most likely next is '{most_common}'. Past success used interventions: {interventions.tolist()}"


# STEP 2: Define sequences and modifiers
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


# STEP 3: Test cases + Planning Suggestion
if __name__ == '__main__':
    engine = SEQCOMEngine()

    # Log test scenarios
    result1 = seqcom_predict("banana_ripening", 2.5, ["peel"])
    result2 = seqcom_predict("code_lifecycle", 6, ["hotfix"])
    result3 = seqcom_predict("focus_decay", 2.2, ["drink_caffeine", "multi_task"])

    engine.log_event("banana_ripening", 2.5, ["peel"], result1)
    engine.log_event("code_lifecycle", 6, ["hotfix"], result2)
    engine.log_event("focus_decay", 2.2, ["drink_caffeine", "multi_task"], result3)

    compressed_df = engine.compress_memory()
    print("\nSEQCOM Compressed Memory Log:")
    print(compressed_df)

    print("\nSuggested Planning Actions:")
    print(engine.suggest_next_action("Brown", "banana_ripening"))
    print(engine.suggest_next_action("Testing", "code_lifecycle"))
    print(engine.suggest_next_action("Engaged", "focus_decay"))
