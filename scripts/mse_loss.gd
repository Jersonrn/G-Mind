class_name MSELoss


var outputs: float = 0.0

var y: Tensor
var y_hat: Tensor


func _init():
	pass


func _ready():
	pass


func forward(y_hat_: Tensor, y_: Tensor) -> Tensor:
	self.y_hat = y_hat_; self.y = y_

	assert(self.y_hat.size() == self.y.size(), "The sizes of 'y_hat' and 'y' must be equal")

	var output:= Tensor.new([self.outputs], self.y_hat.grad_funcs.duplicate())
	var loss: float = 0.0

	for idx in range(len(self.y_hat.values)):
		loss += ( self.y_hat.values[idx] - self.y.values[idx] ) ** 2

	self.outputs = loss / y.size()

	output.values = Array([self.outputs])
	output.add_grad_func(self)

	return output


func calculate_derivative() -> Tensor:
	var output := Tensor.new()

	for idx in range(len(self.y_hat.values)):
		output.append( 2 * (self.y_hat.values[idx] - self.y.values[idx]) )
	
	return output
