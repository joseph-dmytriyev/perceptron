from layer import Layer


class Network:
    
    def __init__(self,input_size,activation):
        
        self.input_size = input_size
        self.activation = activation
        self.layers = []
        
    def add(self,weights,biases):
        
        layer = Layer(weights_list = weights, biases_list = biases)
        self.layers.append(layer)
        
        
    def feedforward(self,inputs):
        
        current_input = inputs
        
        for layer in self.layers:
            
            raw_outputs = layer.forward(current_input)
        
            activated_outputs = [self.activation(output) for output in raw_outputs]
            
            current_input = activated_outputs
            
        return current_input
            