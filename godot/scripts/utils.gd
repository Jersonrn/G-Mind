class_name Utils
extends RefCounted

func linspace(start: float = -1., stop: float = 1., num: int = 100) -> PackedFloat32Array:
	var res: PackedFloat32Array = []
	var step: float = (stop - start) / (num - 1)
	for i in range(num): res.append(start + i * step)

	return res

func is_close(a: float, b: float, rtol: float, atol: float):
	if a == b: return true

	var diff = abs(a - b)
	if diff <= atol: return true

	if diff <= rtol * max(abs(a), abs(b)): return true
	else: return false

func is_all_close(a: PackedFloat32Array, b: PackedFloat32Array, rtol: float =1e-5, atol: float =1e-8):
	assert(len(a) == len(b),"The sizes of 'a' and 'b' must be equal")

	for idx in range(len(a)):
		var x: float = a[idx]
		var y: float = b[idx]

		if not is_close(x, y, rtol, atol):
			return false
	
	return true

func load_data(path: String):
	if not FileAccess.file_exists(path):
		return # Error! We don't have a save to load.

	var data = FileAccess.open(path, FileAccess.READ)
	var json_string = data.get_as_text()

	var json = JSON.new()
	var parse_result = json.parse_string(json_string)

	return parse_result
