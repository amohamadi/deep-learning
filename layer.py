class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None
        self.is_training = True
        
    def forward(self, input_data):
        raise NotImplementedError("forward NotImplementedError")
    
    
    def backward(self, output_error, learning_rate):
        raise NotImplementedError("backward NotImplementedError")
    

