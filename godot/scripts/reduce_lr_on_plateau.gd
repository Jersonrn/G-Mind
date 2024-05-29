class_name ReduceLROnPlateau


var patience:int = 5
var mode:= Modes.MAX
enum Modes {
		MAX,
		MIN
		}
var factor: float = 0.1
var threshold: float = 0.1
var num_bad_epochs: int = 0
var cooldown: int = 0
var min_lr: float = 0.000001
var verbose: bool = false

var cooldown_counter: int = self.cooldown
var best: float = self.worse()

func _init(
		_patience: int = 5,
		_mode: String = "max",
		_factor: float = 0.1,
		_threshold: float = 0.1,
		_num_bad_epochs: int = 0,
		_cooldown: int = 0,
		_min_lr: float = 0.000001,
		_verbose: bool = false,
	):
	self.patience = _patience
	if _mode == "max":
		self.mode = self.Modes.MAX
	elif _mode == "min":
		self.mode = self.Modes.MIN
	else:
		push_error("Unknown mode for scheduler ReduceLROnPlateau, please use 'max' or 'min'")
	self.factor = _factor
	self.threshold = _threshold
	self.num_bad_epochs = _num_bad_epochs
	self.best = self.worse()
	self.cooldown = _cooldown
	self.min_lr = _min_lr
	self.verbose = _verbose

	self.cooldown_counter = self.cooldown


func _ready():
	pass

func reset():
	self.num_bad_epochs = 0
	self.cooldown_counter = 0

func step(metrics, lr) -> float:
	if self.in_cooldown():
		self.cooldown_counter -= 1
	else:
		if self.is_better(metrics):
			self.best = metrics
			self.num_bad_epochs = 0
		else:
			self.num_bad_epochs += 1

	if self.num_bad_epochs > self.patience:
		self.cooldown_counter = self.cooldown
		self.num_bad_epochs = 0

		var new_lr = lr - self.factor

		if new_lr < self.min_lr:
			if verbose:
				print("Min lr reached ", self.min_lr)
			return self.min_lr

		if verbose:
			print("Reducing learning rate to ", new_lr)
		return new_lr
	else:
		return lr

func in_cooldown():
	return self.cooldown_counter > 0

func is_better(current):
	if self.mode == self.Modes.MAX:
		return current > self.best + self.threshold
	elif self.mode == self.Modes.MIN:
		return current < self.best - self.threshold
	else:
		push_error("Unknow mode for scheduler ReduceLROnPlateau")

func worse() -> float:
	if self.mode == self.Modes.MAX:
		return 0.0
	else:
		return 1000000.0

