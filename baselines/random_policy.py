class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, _):
        return [self.action_space.sample()]
    