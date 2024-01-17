extends Node


var xx = [-40., -20.,  0., 20., 40.,  60.,  80.]
var yy = [-40.,  -4., 32., 68., 104., 140., 176.]


var Net = Module.new([
	Dense.new(1, 1),
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

	var num_epochs = 1000.

	for epoch in range(num_epochs):
		for i in range(len(xx)):
			var x: Tensor = Tensor.new([ xx[i] ])
			var y: Tensor = Tensor.new([ yy[i] ])

			var y_hat = Net.forward(x)

			var loss: Tensor = mse_loss.forward(y_hat, y)
			loss.backward()

			Net.step(0.01, true)

			if (epoch + 1) % 100 == 0:
				print("Epoch [", epoch+1./num_epochs, "] Loss: ", loss.values)
	
	var y_t = Tensor.new([75./100.])

	var pred: Tensor = Net.forward(y_t)
	print(pred.values[0]*100.)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func normalize(data: Array) -> Array:
	var outputs: Array = []

	for d in data:
		outputs.append(d/100)

	return outputs
