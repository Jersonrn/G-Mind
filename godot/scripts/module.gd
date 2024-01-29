class_name Module
extends Node


var layers: Array = []


func _init( layers_: Array = [] ):
	self.layers = layers_


func _ready():
	pass


func forward(x: Tensor) -> Tensor:
	var xx: Tensor = x.duplicate()
	
	for layer in self.layers:
		xx.values = layer.forward(PackedFloat32Array(xx.values))
		xx.add_grad_func(layer)

	return xx


func gradients_to_zero():
	for layer in self.layers:
		for c_g_w in layer.gradients_w:
			c_g_w.fill(0)
		layer.gradients_b.fill(0)


#apply_gradients
func step(learn_rate = 0.001, grad_to_zero: bool = false):
	for layer in self.layers:
		layer.apply_gradients(learn_rate, grad_to_zero)
