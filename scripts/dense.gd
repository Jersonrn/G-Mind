class_name Dense


var in_features:  int = 1
var out_features: int = 1

var weights: Array = []
var biases:  Array = []

var inputs := Tensor.new()
var outputs := Tensor.new()

var gradients_w: Array = []
var gradients_b: Array = []


func _init(in_features_: int = 1, out_features_: int = 1, data_layer: Dictionary = {})-> void:
	self.in_features  = in_features_
	self.out_features = out_features_

	if data_layer:
		print("Layer set data not implemented yet...")
	else:
		self.gen_randf_weights_bias_and_zero_gradients()


func _ready():
	pass


func gen_randf_weights_bias_and_zero_gradients() -> void:
	randomize()

	for node_out_index in range(self.out_features):
		var node_out_weights: Array = []

		var row_gradients: Array = []

		for in_feature_index in range(self.in_features):
			node_out_weights.append(randf_range(-1, 1))

			row_gradients.append(0.0)

		self.weights.append(node_out_weights)
		self.biases.append(randf_range(-1, 1))

		self.gradients_w.append(row_gradients)
		self.gradients_b.append(0.0)


func forward(x: Tensor) -> Tensor:
	assert(x.size() == self.in_features,"Error: The size of the input data doesn't match the expected input features for the layer.")

	self.inputs = x
	self.outputs.values = []

	for node_weights_index in range(len(self.weights)):
		var node_weights: Array = self.weights[node_weights_index]

		var node_output: float = 0.0

		for weight_index in range(len(node_weights)):
			var weight: float = node_weights[weight_index]

			node_output += x.values[weight_index] * weight

		node_output += self.biases[node_weights_index]

		self.outputs.append(node_output)

	var xx := Tensor.new(self.outputs.values, x.grad_funcs)
	xx.add_grad_func(self)

	return xx


func calculate_derivative() -> Tensor:
	return Tensor.new(self.weights)


func derivative_respect_inputs()-> Tensor:
	return Tensor.new(self.weights)


func derivative_respect_weights() -> Tensor:
	return self.inputs


func apply_gradients(learn_rate: float = 0.001, grad_to_zero: bool = true ):
	for node_out in range(self.out_features):
		for node_in in range(self.in_features):
			self.weights[node_out][node_in] -= ( self.gradients_w[node_out][node_in] * learn_rate ) * 0.5

		self.biases[node_out] -= ( self.gradients_b[node_out] * learn_rate ) * 0.5

		if grad_to_zero:
			self.gradients_w[node_out].fill(0)

	if grad_to_zero:
		self.gradients_b.fill(0)
