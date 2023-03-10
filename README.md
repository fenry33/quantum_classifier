# Quantum_Classifier
Just test.
Newbie in quantum computing. Code with pennylane and tensorflow. Thank you.

# files
Use quantum classifier to classify mnist data.
mnist.py prepocesses data with binary label 0 or 1.
amplitude_encoding.py builds quantum model with amplitude encoding for classifying.
classicl_classifier.py builds just a simple NN model.
hybrid_classifier.py provides hybrid quantum-classical model and compares its result with classical NN model.
Also images of results are provided.

# problems
Training results of quanumr circuit with amplitude encoding gives unchanged accuracy for both training data and validation data. I dont know how to address it.
Parameters in hybrid model and classsical NN are almost the same but i dont understand how to evaluate parameters in quantum circuit and networks.
