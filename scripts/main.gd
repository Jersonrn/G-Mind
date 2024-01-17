extends Node


var x = Tensor.new([8, 9])
var y = Tensor.new([3])

var Net = Module.new([
	Dense.new(2, 3),
	Dense.new(3, 1),
	])

var mse_loss = MSELoss.new()

# Called when the node enters the scene tree for the first time.
func _ready():
	self.Net.layers[0].weights = [[-0.86856418376318, 0.70221519908339], [0.89603692622963, -0.73766922689399], [-0.05890860177168, -0.88088801462823]]
	self.Net.layers[0].biases =  [-0.41828938217136, -0.61897680519048, 0.1618861589]

	self.Net.layers[1].weights = [[0.38369935275339, -0.97945182189683, -0.79472319837192]]
	self.Net.layers[1].biases = [0.08792996174267]

	self.do()



func do():
	var y_hat: Tensor = Net.forward(x)

	for layer in self.Net.layers:
		print("L_W: ", layer.weights)
		print("L_B: ", layer.biases)
		print("--------------------------")

	var loss: Tensor = mse_loss.forward(y_hat, y)
	print("loss: ", loss.values)
	print("loss_G : ", loss.grad_funcs)

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
