extends Node


var x = Tensor.new([8, 9])
var y = Tensor.new([3])

var Net = Module.new([
	Dense.create(2, 3),
	Dense.create(3, 1),
	])

var mse_loss = MSELoss.new()

func _ready():
	self.do()



func do():
	var y_hat: Tensor = Net.forward(x)

	for layer in self.Net.layers:
		print("L_W: ", layer.weights)
		print("L_B: ", layer.biases)
		print("--------------------------")

	var loss: Tensor = mse_loss.forward(y_hat, y)
	print("loss: ", loss.values)
	# print("loss_G : ", loss.grad_funcs)
	print("--------------------------")

	loss.backward()
	for layer in self.Net.layers:
		print("GW: ", layer.gradients_w)
		print("GB: ", layer.gradients_b)
		print("--------------------------")
	print("--------------------------")

	Net.step(0.001, true)
	for layer in self.Net.layers:
		print("L_W: ", layer.weights)
		print("L_B: ", layer.biases)
		print("--------------------------")



# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
