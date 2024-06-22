class_name Utils
extends RefCounted

func linspace(start: float = -1, stop: float = 1, num: int = 100) -> Array:
	var res: Array = []
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
