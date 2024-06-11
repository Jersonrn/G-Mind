class_name  Sigmoid


var inputs : PackedFloat32Array

func _init(inputs_: PackedFloat32Array = []):
	self.inputs = inputs_


func _ready():
	pass


func _to_string():
	return "Sigmoid()".format({})


func forward(xx: PackedFloat32Array) -> PackedFloat32Array:
	self.inputs = xx
	var output: PackedFloat32Array = []

	for x  in xx: output.append( 1 / (1 + exp(-x)) )

	return output


func calculate_derivative() -> Tensor:
	var output := Tensor.new()

	for x in self.inputs:
		output.append(exp(-x) * (1 + exp(-x)) ** -2)

	return  output

func save():
	var data = {
			"type": "Sigmoid",
			"inputs": self.inputs
			}

	return data
