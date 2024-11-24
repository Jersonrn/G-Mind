class_name MinMaxScaler

var utl = Utils.new()

var min_value = 0
var max_value = 0

func _init():
	pass

func _ready():
	pass

func fit(data: Tensor):
	var min_max_values = self.utl.min_max(data.values)
	self.min_value = min_max_values[0]
	self.max_value = min_max_values[1]

func transform(data: Tensor):
	var outputs = Tensor.new([])

	for x in data.values:
		var r = (float(x) - self.min_value) / (self.max_value - self.min_value)
		outputs.append(r)

	return outputs

func inverse_transform(data: Tensor):
	var outputs = Tensor.new([])

	for x in data.values:
		var r = (float(x) * (self.max_value - self.min_value) + self.min_value)
		outputs.append(r)

	return outputs
