class_name Plotter
extends Control


@export var width: int = 0
@export var height: int = 0
@export var expand: int = 1

@export var cartesian_coordinate = Vector2(0., 0.)

var data: Array = []
@onready var rect := Rect2(
		Vector2(self.position),
		Vector2(self.size)
		)


func _init(_width: int = 0, _height: int = 0, _expand: int = 1):
	self.width = _width; self. height = _height; expand = _expand


func _ready():
	self.cartesian_coordinate = self.position + (self.cartesian_coordinate*self.expand)
	self.initialize_with_zeros()
	self.queue_redraw()


func initialize_with_zeros() -> void:
	var rgba: Array = [0.0, 0.0, 0.0, 0.0]
	
	for h in range(self.height):
		var x: Array = []

		for w in range(self.width):
			x.append(rgba.duplicate())

		self.data.append(x)
		
	self.queue_redraw()


func _normalize(values: Array) -> Array:
	var output: Array = []
	for v in values: output.append((v + 1) / 2)

	return output


func _draw() -> void:
	for y in range(self.height):
		for x in range(self.width):
			var color: Array = self.data[y][x]
			var r = 0; var g = 1; var b = 2; var a = 3

			draw_rect(
					# Rect2(x, y, 1, 1),
					Rect2(x*expand, y*expand, expand, expand),
					Color(color[r], color[g], color[b], color[a])
					)
