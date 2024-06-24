#!/usr/bin/env -S godot -s
extends SceneTree



var utl = Utils.new()
var data: Dictionary

var n_passed: int = 0
var n_failed: int = 0


func _init():
	# print(1.00001e10, PackedFloat32Array([1.00001e10]))

	data = utl.load_data("res://test/data/data.json")
	
	self.test_sigmoid(
			data["PackedFloat32Array"],
			data["Sigmoid"],
			data["SigmoidDeriv"]
		)

	self.print_result()
	quit()


func test_create_sigmoid() -> bool:
	var sigmoid = Sigmoid.new()

	return true


func test_forward_sigmoid(inputs: PackedFloat32Array, expected: PackedFloat32Array) -> bool:
	var sigmoid = Sigmoid.new()

	var result = sigmoid.forward(inputs)

	return utl.all_close(expected, result, 1e-04, 1e-07)


func test_derivative_sigmoid(inputs: PackedFloat32Array, expected: PackedFloat32Array) -> bool:
	var sigmoid = Sigmoid.new(inputs)

	var result = sigmoid.calculate_derivative().values

	return utl.all_close(expected, result, 1e-04, 1e-07)




func test_sigmoid(
		inputs: PackedFloat32Array,
		expected_sigmoid: PackedFloat32Array,
		expected_deriv_sigmoid: PackedFloat32Array
	):
	#create
	var create = self.test_create_sigmoid()
	self.show_info("Sigmoid.create()", create)

	#forward
	var forward = self.test_forward_sigmoid(inputs, expected_sigmoid)
	self.show_info("Sigmoid.forward()", forward)

	#deriv
	var deriv = self.test_derivative_sigmoid(inputs, expected_deriv_sigmoid)
	self.show_info("Sigmoid.calculate_derivative()", deriv)











func print_green_line(msg: String = "") -> void:
	Print.new().line(Print.new().GREEN, msg)

func print_yellow_line(msg: String = "") -> void:
	Print.new().line(Print.new().YELLOW, msg)

func print_red_line(msg: String = "") -> void:
	Print.new().line(Print.new().RED, msg)

func print_result():
	if n_failed > 0 and n_passed > 0:
		print_yellow_line("======================================")
		print_yellow_line("{failed} failed and {passed} passed: total = {total}".format({"failed": self.n_failed, "passed": self.n_passed, "total": self.n_passed + self.n_failed}))
		print_yellow_line("--------------------------------------")
		print_yellow_line("               END TEST               ")
		print_yellow_line("======================================")

	elif n_passed == 0:
		print_red_line("======================================")
		print_red_line("{failed} failed and {passed} passed: total = {total}".format({"failed": self.n_failed, "passed": self.n_passed, "total": self.n_passed + self.n_failed}))
		print_red_line("--------------------------------------")
		print_red_line("               END TEST               ")
		print_red_line("======================================")

	elif n_failed == 0:
		print_green_line("======================================")
		print_green_line("{failed} failed and {passed} passed: total = {total}".format({"failed": self.n_failed, "passed": self.n_passed, "total": self.n_passed + self.n_failed}))
		print_green_line("--------------------------------------")
		print_green_line("               END TEST               ")
		print_green_line("======================================")



func show_info(name_func: String, passed: bool) -> void:
	if passed:
		self.n_passed += 1
		var msg = name_func + " passed!"

		print_green_line("--------------------------------------")
		print_green_line(msg)
		print_green_line("--------------------------------------")

	else:
		self.n_failed += 1
		var msg = name_func + " failed!"

		print_red_line("--------------------------------------")
		print_red_line(msg)
		print_red_line("--------------------------------------")
