extends Node

var new_weights=[[-0.34063065, -0.8958533], [0.76431787, -0.82561517], [0.9294969, 0.24996507]]
var weights = Array([ 
	PackedFloat32Array([-0.10176134, 0.9617566]),
	PackedFloat32Array([-0.9623445, -0.66758066]),
	PackedFloat32Array([-0.9623445, -0.66758066]),
	PackedFloat32Array([-0.06156087, -0.7936802]),
	])
var biases = PackedFloat32Array([0.037294745, -0.07986778, -0.32146078])

var dense = Dense.create(2, 3)


var x := Array([8., 9.])
var y := PackedFloat32Array([8., 9.])
var z := Array([[2., 3.]])
var g := Tensor.new([2., 4.])


func _ready():
	print(g.values)
	g.values = dense.forward(PackedFloat32Array(g.values))
	print(g.values)
	#print(sequential)
	#sequential.forward()
	#print("----------------------")
	#print(sequential)
	





# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
