from data.initial_data import *

class ExperimentPhase:

    def __init__(self, phase_number: int, before_interaction: bool):
        self.phase_number = phase_number
        self.initial_class_assignement = initial_classes(phase_number)
        self.initial_graph = initial_graph(phase_number)
        self.before_interaction = before_interaction