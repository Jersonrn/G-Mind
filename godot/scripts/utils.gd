extends Node

func linspace(start: float = -1, stop: float = 1, num: int = 100) -> Array:
	var res: Array = []
	var step: float = (stop - start) / (num - 1)
	for i in range(num): res.append(start + i * step)

	return res
