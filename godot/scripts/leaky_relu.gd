class_name LeakyRelu


@export var negative_slope: float = 0.01

var inputs : PackedFloat32Array


func _init(negative_slope_: float = 0.01):
	self.negative_slope = negative_slope_


func _ready():
	pass


func _to_string():
	return "LeakyRelu(negative_slope={negative_slope})".format({"negative_slope":self.negative_slope})


func forward(xx: PackedFloat32Array) -> PackedFloat32Array:
	self.inputs = xx
	var output: PackedFloat32Array = []

	for x in xx: output.append(max(self.negative_slope * x, x))

	return output


func calculate_derivative() -> Tensor:
	var output := Tensor.new()

	for x in self.inputs:
		if x >= 0:
			output.append(1.0)
		else:
			output.append(self.negative_slope)

	return output

