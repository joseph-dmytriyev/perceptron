class Neuron:
  
    
    def __init__(self, weights, bias):       
        self.weights = weights
        self.bias = bias
    
    def forward(self, inputs):
       
        
        output = sum(w * x for w, x in zip(self.weights, inputs))
        
        output += self.bias
        return output