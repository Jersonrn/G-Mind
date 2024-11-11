extends Node


var xx = [-40., -20.,  0., 20., 40.,  60.,  80.]#Celsius
var yy = [-40.,  -4., 32., 68., 104., 140., 176.]#Fahrenheit

var seed = hash("G-Mind")

var Net = Sequential.new([
	Dense.create(1, 1, self.seed),
	])


var mse_loss = MSELoss.new()


# Called when the node enters the scene tree for the first time.
func _ready():
	xx = [-0.4, -0.2,  0.,   0.2,  0.4,  0.6,  0.8]
	yy = [-0.4,  -0.04,  0.32,  0.68,  1.04,  1.4,   1.76]

	# xx = self.normalize(xx)
	# yy = self.normalize(yy)
	print(xx)
	print(yy)
	print("---------------------------------")

func train():
	var epochs = 1000.
	var batch_size = 1
	var learning_rate = 0.01

	var count = 0

	for epoch in range(epochs):
		var epoch_loss = 0.

		for i in range(len(xx)):
			var x: Tensor = Tensor.new([ xx[i] ])
			var y: Tensor = Tensor.new([ yy[i] ])

			var y_hat = Net.forward(x)

			var loss: Tensor = mse_loss.forward(y_hat, y)
			epoch_loss += loss.values[0]
			loss.backward(batch_size)

			count += 1

			Net.clip_gradients(1.0)

			if count % batch_size == 0:
				Net.step(learning_rate, true)


		if (epoch + 1) % 10 == 0:
			print("Epoch [", epoch + 1., "] Loss: ", epoch_loss / len(xx))
	

func do_pred():
	var x = Tensor.new([75./100.])#Celsius

	var pred: Tensor = Net.forward(x)

	print("X = {x_values} -- Y = 167 -- Y_HAT = {y_hat_values}".format(
		{ "x_values": x.values[0] * 100., "y_hat_values": pred.values[0] * 100. }
		))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func normalize(data: Array) -> Array:
	var outputs: Array = []

	for d in data:
		outputs.append(d/100)

	return outputs


func _on_celsius_input_text_changed():
	var celsius = $calculator_container/celsius_container/celsius_input.text
	celsius = float(celsius)/100.
	
	var x = Tensor.new([ float(celsius) ])#Celsius

	var pred: Tensor = Net.forward(x)
	var y = pred.values[0] * 100.
	
	$calculator_container/fahrenheit_container/n_fahrenheit.text = str(y)


func _on_train_pressed():
	self.train()
	$model_container/train.toggle_mode = false
