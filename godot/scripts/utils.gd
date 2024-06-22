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
