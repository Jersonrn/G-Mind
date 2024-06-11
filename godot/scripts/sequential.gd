class_name Sequential
extends Node


var layers: Array = []


func _init( layers_: Array = [] ):
	self.layers = layers_


func _ready():
	pass

func _to_string():
	return "Sequential(layers={layers})".format({"layers":layers})

func forward(x: Tensor) -> Tensor:
	var xx: Tensor = x.duplicate()
	
	for layer in self.layers:
		xx.values = layer.forward(PackedFloat32Array(xx.values))
		xx.add_grad_func(layer)

	return xx

func clip_gradients(max_norm: float = 1.0):
	var norm = 0.

	for layer in layers:
		if "gradients_w" in layer:
			norm += layer.calculate_gradient_norm()
	
	if norm > max_norm:
		var factor = max_norm / norm

		for layer in layers:
			if "gradients_w" in layer:
				layer.normalize_gradients(factor)


func gradients_to_zero():
	for layer in self.layers:
		if "gradients_w" in layer:
			layer.gradients_2_zero()

func save_model(path: String = "res://models/model.gmind"):
	var save_model = FileAccess.open(path, FileAccess.WRITE)

	for idx in range(len(self.layers)):
		var layer = self.layers[idx]
		var layer_data = layer.save()

		var json_string = JSON.stringify(layer_data)

		save_model.store_line(json_string)


func load_model(path: String = "res://models/model.gmind"):
	if not FileAccess.file_exists(path):
		return # Error! We don't have a save to load.

	var save_model = FileAccess.open(path, FileAccess.READ)

	while save_model.get_position() < save_model.get_length():
		var json_string = save_model.get_line()

		# Creates the helper class to interact with JSON
		var json = JSON.new()

		# Check if there is any error while parsing the JSON string, skip in case of failure
		var parse_result = json.parse(json_string)
		if not parse_result == OK:
			print("JSON Parse Error: ", json.get_error_message(), " in ", json_string, " at line ", json.get_error_line())
			continue

		# Get the data from the JSON object
		var layer_data = json.get_data()

		if layer_data["type"] == "Dense":
			var in_features = int(layer_data["in_features"])
			var out_features = int(layer_data["out_features"])

			var weights: Array[PackedFloat32Array]
			for packedf32array in layer_data["weights"]:
				weights.append(PackedFloat32Array(packedf32array))

			var biases: PackedFloat32Array = layer_data["biases"]

			var gradients_w: Array[PackedFloat32Array]
			for packedf32array in layer_data["gradients_w"]:
				gradients_w.append(PackedFloat32Array(packedf32array))

			var gradients_b: PackedFloat32Array = layer_data["gradients_b"]
			

			self.layers.append(
					Dense.from_data(
						in_features,
						out_features,
						weights,
						biases,
						gradients_w,
						gradients_b,
						)
					)
		elif layer_data["type"] == "Sigmoid":
			var inputs: PackedFloat32Array = layer_data["inputs"]

			self.layers.append(
					Sigmoid.new(
						inputs
						)
					)
		elif layer_data["type"] == "LeakyRelu":
			var inputs: PackedFloat32Array = layer_data["inputs"]
			var negative_slope: float = layer_data["negative_slope"]

			self.layers.append(
					LeakyRelu.new(
						negative_slope,
						inputs
						)
					)
	

#apply_gradients
func step(learn_rate = 0.001, grad_to_zero: bool = false):
	for layer in self.layers:
		if "gradients_w" in layer:
			layer.apply_gradients(learn_rate, grad_to_zero)
