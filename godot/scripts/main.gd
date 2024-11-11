extends Node

var seed = hash("G-Mind")

var x = Tensor.new([8, 9])
var y = Tensor.new([3])

var w: Array[PackedFloat32Array] = [
			[0.25473, 0.82283795],
			[-0.67069113, -0.8697771],
			[0.7895458, -0.51921535]
			]
var b: PackedFloat32Array = [-0.4931361, 0.9203217, 0.89365923]

var gw: Array[PackedFloat32Array] = [
			[0., 0.],
			[0., 0.],
			[0., 0.]
			]
var gb: PackedFloat32Array = [0., 0., 0.]


var Net = Sequential.new([
	Dense.create(3, 1, self.seed),
	Sigmoid.new(),
	Dense.from_data(2, 3, w, b, gw, gb),
	LeakyRelu.new(0.1)
	])


var mse_loss = MSELoss.new()

func _ready():
	Net.save_model()
	print(Net)
	print("***********************************")

	Net.layers = []
	Net.load_model()
	print(Net)

	# self.do()
	pass



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

	loss.backward(1)
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
