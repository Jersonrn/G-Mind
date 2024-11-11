extends Control

@onready var void_ = $Void_
@onready var left_panel = $left_panel
@onready var left_panel_rect = Rect2(
		Vector2(left_panel.position),
		Vector2(left_panel.size)
	)
@onready var brush = $left_panel/tools/brush
@onready var erase = $left_panel/tools/erase
@onready var class_selector = $left_panel/class_selector
@onready var reds = $classes/reds
@onready var blues = $classes/blues
@onready var plotter = $Plotter


@export var red_scene: PackedScene
@export var blue_scene: PackedScene

@export var tool_sel:= Tools.NONE
enum Tools {
		NONE,
		BRUSH,
		ERASE
	}

var dragging = false

var seed = hash("G-Mind")

var model = Sequential.new([
	Dense.create(2, 15, self.seed),
	LeakyRelu.new(),
	Dense.create(15, 15, self.seed),
	LeakyRelu.new(),
	Dense.create(15, 15, self.seed),
	LeakyRelu.new(),
	Dense.create(15, 15, self.seed),
	LeakyRelu.new(),
	Dense.create(15, 15, self.seed),
	LeakyRelu.new(),
	Dense.create(15, 1, self.seed),
	LeakyRelu.new(),
	])

var mse_loss = MSELoss.new()

var scheduler = ReduceLROnPlateau.new(
		5,			#patience
		"min",		#mode
		0.001,		#factor
		0.0001,		#threshold
		0,			#num_bad_epochs
		0,			#cooldown
		0.000001,	#min_lr
		true		#verbose
	)

var epochs = 1
var batch_size = 1
var learning_rate = 0.1

var training = false 
var do_pred = false 


func _ready():
	self.do_prediction()
	# var result = OS.execute("scrot", ["/home/jersonrn/Pictures/screenshots_binary/%Y-%m-%d_%H-%M-%S.png"])
	# if result == OK:
	# 	print("todo bien")
	# else:
	# 	print("todo mal")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	if training:
		train()

func point_to_plotter(plotter: Plotter, point: Vector2) -> Vector2:
	return (point - plotter.cartesian_coordinate) / plotter.expand

func points_to_plotter(plotter_: Plotter, points: Array) -> Array:
	var output: Array = []

	for point in points:
		output.append( (point.position - plotter.cartesian_coordinate) / plotter.expand )

	return output

func train():
	var start_time: float = Time.get_ticks_msec()
	var epoch_loss = 0
	print("=================================================================")

	for point in self.reds.get_children():
		var point_pos = self.point_to_plotter(self.plotter, point.position)
		var x = Tensor.new([ point_pos.x/plotter.width, point_pos.y/plotter.height ])
		var y = Tensor.new([ 0. ])

		var y_hat = self.model.forward(x)
		# print(x, " --- ", y, " --- ", y_hat)
		
		var loss: Tensor = mse_loss.forward(y_hat, y)
		epoch_loss += loss.values[0]
		loss.backward(batch_size)

		model.clip_gradients(1.0)
		model.step(learning_rate, true)
	print("----------------------------------------------------------")

	for point in self.blues.get_children():
		var point_pos = self.point_to_plotter(self.plotter, point.position)
		var x = Tensor.new([ point_pos.x/plotter.width, point_pos.y/plotter.height ])
		var y = Tensor.new([ 1. ])

		var y_hat = self.model.forward(x)
		# print(x, " --- ", y, " --- ", y_hat)
		
		var loss: Tensor = mse_loss.forward(y_hat, y)
		epoch_loss += loss.values[0]
		loss.backward(batch_size)

		model.clip_gradients(1.0)
		model.step(learning_rate, true)

	var end_time: float = Time.get_ticks_msec()
	var elapsed_time: float = end_time - start_time

	var loss_avg = epoch_loss

	print("epoch_loss = ", loss_avg, " -- elapsed_time = ", elapsed_time/1000, " seconds...")

	self.learning_rate = scheduler.step(loss_avg, self.learning_rate)

	self.do_prediction()
	# self.training = false
	var result = OS.execute("scrot", ["/home/jersonrn/Pictures/screenshots_binary/%Y-%m-%d_%H-%M-%S.png"])

	# var capture = get_viewport().get_texture().get_image()
	# var _time = Time.get_datetime_string_from_system()
	# var filename = "res://screenshots/Screenshot-{0}.png".format({"0":_time})
	# capture.save_png(filename)

func do_prediction():
	for h in range(self.plotter.height):
		for w in range(self.plotter.width):
			var in_vector = Vector2(float(w), float(h)) - Vector2(50., 50.)
			var x := Tensor.new([in_vector.x/float(self.plotter.width), in_vector.y/float(self.plotter.height)])
			var pred: Tensor = self.model.forward(x)
			# print(x, " --- ", pred)
			var rgba := []
			if pred.values[0] >= 0.5:
				rgba = [0.07, 0.14, 0.8, 1.]
			elif pred.values[0] < 0.5:
				rgba = [0.8, 0.07, 0.14, 1.]

			self.plotter.data[h][w] = rgba
	
	self.plotter.queue_redraw()


func _input(event):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		# Start dragging
		if not self.dragging and event.pressed:
			self.dragging = true

			if tool_sel == Tools.ERASE:
				if self.plotter.rect.has_point(event.position):
					var childrens = self.reds.get_children() + self.blues.get_children()

					for child in childrens:
						if self.void_.get_rect().has_point(self.void_.to_local(child.position)):
							child.queue_free()
				
		# Stop dragging if the button is released.
		if self.dragging and not event.pressed:
			self.dragging = false

			if self.tool_sel == self.Tools.BRUSH:
				if self.plotter.rect.has_point(event.position):
					if class_selector.get_selected_id() == 0:
						var red = red_scene.instantiate()
						red.position = event.position
						self.reds.add_child(red)

					elif self.class_selector.get_selected_id() == 1:
						var blue = blue_scene.instantiate()
						blue.position = event.position
						self.blues.add_child(blue)

	elif event is InputEventMouseMotion:
		self.void_.position = event.position


func _on_brush_toggled(toggled_on):
	if toggled_on:
		self.erase.button_pressed = false
		self.tool_sel = Tools.BRUSH
	else:
		tool_sel = Tools.NONE


func _on_erase_toggled(toggled_on):
	if toggled_on:
		self.brush.button_pressed = false
		self.tool_sel = Tools.ERASE
	else:
		tool_sel = Tools.NONE


func _on_train_toggled(toggled_on):
	self.training = toggled_on
