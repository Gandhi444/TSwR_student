import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        e=(q_r-[q1,q2])[:, np.newaxis]
        e_dot=(q_r_dot-[q1_dot,q2_dot])[:, np.newaxis]
        Kd = np.array([[25, 0], [0, 25]])
        Kp = np.array([[25, 0], [0, 25]])
        v=q_r_ddot[:, np.newaxis]+Kd@e_dot+Kp@e
        tau=self.model.M(x) @ v + self.model.C(x)@np.array([[q1_dot],[q2_dot]])
        return tau
