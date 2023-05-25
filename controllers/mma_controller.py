import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        M1=ManiuplatorModel(Tp)
        M1.m3=0.1
        M1.r3=0.05
        M2=ManiuplatorModel(Tp)
        M2.m3=0.01
        M2.r3=0.01
        M3=ManiuplatorModel(Tp)
        M3.m3=1.0
        M3.r3=0.3

        self.models = [M1, M2, M3]
        self.i = 0
        self.u = np.zeros((2, 1))
    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q1, q2, q1_dot, q2_dot = x
        errors=[]
        for model in self.models:
            y=model.M(x)@self.u+model.C(x)@[[q1_dot],[q2_dot]]
            err=np.sum(abs([[q1],[q2]]-y))
            errors.append(err)
        self.i=np.argmin(errors)
    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        q1, q2, q1_dot, q2_dot = x
        Kd = np.array([[25, 0], [0, 25]])
        Kp = np.array([[25, 0], [0, 25]])
        e=(q_r-[q1,q2])[:, np.newaxis]
        e_dot=(q_r_dot-[q1_dot,q2_dot])[:, np.newaxis]
        v=q_r_ddot[:, np.newaxis]+Kd@e_dot+Kp@e
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot[:, np.newaxis]
        self.u=u
        return u
