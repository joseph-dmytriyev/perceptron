from neuron import Neuron


class Layer:
    
    def __init__(self,weights_list,biases_list):
        
        self.neurons = []
        
        for weights, bias in zip(weights_list,biases_list):
            neuron = Neuron(weights = weights , bias = bias)
            self.neurons.append(neuron)
            
    def forward(self, inputs):
        
        outputs = []
        
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            outputs.append(output)
            
        return outputs