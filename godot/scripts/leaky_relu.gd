class_name LeakyRelu


@export var alpha: float = 0.01

var inputs : PackedFloat32Array


func _init(alpha_: float = 0.01):
	self.alpha = alpha_


func _ready():
	pass


func _to_string():
	return "LeakyRelu(alpha={alpha})".format({"alpha":self.alpha})


func forward(xx: PackedFloat32Array) -> PackedFloat32Array:
	self.inputs = xx
	var output: PackedFloat32Array = []

	for x in xx: output.append(max(self.alpha * x, x))

	return output


func calculate_derivative() -> Tensor:
	var output := Tensor.new()

	for x in self.inputs:
		if x >= 0:
			output.append(1.0)
		else:
			output.append(self.alpha)

	return output

