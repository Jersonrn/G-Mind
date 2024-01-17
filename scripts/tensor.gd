class_name Tensor


var values: Array = []
var grad_funcs: Array = []


func _init(values_: Array = [], grad_funcs_: Array = []):
	self.values = values_
	self.grad_funcs = grad_funcs_


func add_grad_func(grad_func):
	self.grad_funcs.append(grad_func)


func append(x):
	self.values.append(x)


func backward():
	assert(!self.grad_funcs.is_empty(),"No 'grad_funcs' found for this Tensor")

	var output: Tensor = self.grad_funcs[-1].calculate_derivative().ones_like()

	for idx in range(len(self.grad_funcs) - 1, -1, -1):
		var current_layer = self.grad_funcs[idx]

		if "gradients_w" in current_layer:
			#Updating gradients
			var derivative_weights: Tensor = current_layer.derivative_respect_weights()

			for n_o_idx in range(current_layer.out_features):
				for n_i_idx in range(current_layer.in_features):
					current_layer.gradients_w[n_o_idx][n_i_idx] += output.values[n_o_idx] * derivative_weights.values[n_i_idx]

				current_layer.gradients_b[n_o_idx] += 1 * output.values[n_o_idx]

			#Backpropagation
			#The length of the weights for each node in the current layer (len(self.weights[0 | 1 | ...n]))
			#matches the number of output nodes from the prev layer, which is precisely what we require at this point.
			var n_nodes_out_prev_layer = len(current_layer.weights[0])

			var derivative_inputs: Tensor = current_layer.derivative_respect_inputs()

			var new_total_derivative: Array = []

			for n_o_p_idx in range(n_nodes_out_prev_layer):
				var new_node_derivative : float = 0.0

				for n_o_c_idx in range(current_layer.out_features):
					var weight = derivative_inputs.values[n_o_c_idx][n_o_p_idx]
					var node_derivative = output.values[n_o_c_idx]

					new_node_derivative +=  weight * node_derivative 

				new_total_derivative.append(new_node_derivative)

			output.values = new_total_derivative
		else:
			var derivative_layer: Tensor = current_layer.calculate_derivative()
			output = output.multiply(derivative_layer)


func clear():
	self.values.clear()
	self.grad_funcs.clear()


func clear_grad_funcs():
	self.grad_funcs.clear()


func duplicate():
	return Tensor.new(self.values.duplicate(), self.grad_funcs.duplicate())


func get_shape(list = self.values):
	var shape = []
	
	while list is Array:
		shape.append(list.size())
		list = list[0] if list.size() > 0 else null

	return shape


func multiply(b: Tensor) -> Tensor:
	assert(self.size() == b.size(), "The sizes of 'a' and 'b' must be equal")

	var outputs := Tensor.new()

	for idx in range(len(self.values)):
		outputs.append( self.values[idx] * b.values[idx] )

	return outputs


func size():
	return len(self.values)


func ones_like() -> Tensor:
	var val: Array = []
	val.resize( self.size() )
	val.fill(1.)
	return Tensor.new(val)


func zeros_like() -> Tensor:
	var val: Array = []
	val.resize( self.size() )
	val.fill(0)
	return Tensor.new(val)
