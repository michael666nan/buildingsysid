class StateSpace:
    
    def __init__(self, A, B, C, D, K=None, x0=None, samplingTime = 0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = K
        self.x0 = x0
    
        self.samplingTime = samplingTime       #0 = continuous time, otherwise discrete