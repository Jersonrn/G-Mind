class_name Utils
extends RefCounted

func linspace(start: float = -1., stop: float = 1., num: int = 100) -> PackedFloat32Array:
	var res: PackedFloat32Array = []
	var step: float = (stop - start) / (num - 1)
	for i in range(num): res.append(start + i * step)

	return res

func is_close(a: float, b: float, rtol: float =1e-05, atol: float =1e-08):
	if abs(a - b) <= (atol + rtol * abs(b)):
		return true
	else:
		return false

func all_close(a: PackedFloat32Array, b: PackedFloat32Array, rtol: float =1e-05, atol: float =1e-08):
	assert(len(a) == len(b),"The sizes of 'a' and 'b' must be equal")

	for idx in range(len(a)):
		var x: float = a[idx]
		var y: float = b[idx]
		print(idx, " -- ",x, " -- ", y)

		if not abs(x - y) <= (atol + rtol * abs(y)):
			return false
	
	return true

func load_data(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}

	var data = FileAccess.open(path, FileAccess.READ)
	var json_string = data.get_as_text()

	var json = JSON.new()
	var parse_result = json.parse_string(json_string)

	return parse_result

func save_data(path: String, what: Dictionary, append: bool = false):
	var data: Dictionary = {}

	if append:
		data = self.load_data(path)

	for i in what:
		data[i] = what[i]

	var save_data = FileAccess.open(path, FileAccess.WRITE)


	var json_string = JSON.stringify(data)
	save_data.store_line(json_string)
