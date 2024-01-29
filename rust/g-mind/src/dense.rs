use godot::prelude::*;
use godot::engine::{IRefCounted, RandomNumberGenerator};
// use crate::tensor::Tensor;


#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct  Dense {
    #[var]
    in_features: i8,
    #[var]
    out_features: i8,

    #[var]
    inputs: PackedFloat32Array,
    #[var]
    outputs: PackedFloat32Array,

    #[var]
    weights: Array<PackedFloat32Array>,
    #[var]
    biases: PackedFloat32Array,

    #[var]
    gradients_w: Array<PackedFloat32Array>,
    #[var]
    gradients_b: PackedFloat32Array,

    #[base]
    base: Base<RefCounted>
}

#[godot_api]
impl IRefCounted for Dense {
    fn init(base: Base<RefCounted>) -> Self {
        let mut dense_instance = Self {
            in_features: 1,
            out_features: 1,

            inputs: PackedFloat32Array::new(),
            outputs: PackedFloat32Array::new(),

            weights: array![],
            biases: PackedFloat32Array::new(),

            gradients_w: array![],
            gradients_b: PackedFloat32Array::new(),

            base,
        };

        dense_instance //return
    }

    fn to_string(&self) -> GString {
        let Self { in_features, out_features, weights, biases, .. } = &self;
        format!("Dense(
                in_features={in_features},
                out_features={out_features},
                weights={weights},
                biases={biases})",
            ).into()
    }

}

#[godot_api]
impl Dense {
    #[func]
    fn apply_gradients(&mut self, lr: f32, clean_grad: bool) {
        let mut new_layer_weights: Array<PackedFloat32Array> = array![];
        let mut new_layer_biases = PackedFloat32Array::new();

        for node_out_index in 0..self.weights.len() {
            let node_weights: PackedFloat32Array = self.weights.get(node_out_index);

            let mut new_node_gradients_w = PackedFloat32Array::new();
            let mut new_node_gradients_b = 0.;

            for node_in_index in 0..node_weights.len() {
                let weight = self.weights.get(node_out_index).get(node_in_index);
                let gradient_w = self.gradients_w.get(node_out_index).get(node_in_index);

                new_node_gradients_w.push( weight - (gradient_w * lr) );
            };

            let bias = self.biases.get(node_out_index);
            let gradient_b = self.gradients_b.get(node_out_index);

            new_node_gradients_b = bias - (gradient_b * lr);


            new_layer_weights.push(new_node_gradients_w);
            new_layer_biases.push(new_node_gradients_b);

            if clean_grad == true {
                self.gradients_w.get(node_out_index).fill(0.)
                
            }
        };

        self.weights = new_layer_weights;
        self.biases = new_layer_biases;

        if clean_grad == true {
            self.gradients_b.fill(0.);
        }

    }

    #[func]
    fn forward(&mut self, x:PackedFloat32Array,) -> PackedFloat32Array {
        assert!(x.len() == self.in_features as usize, "Error: The size of the input data doesn't match the expected input features for the layer.");

        self.inputs = x;
        self.outputs = PackedFloat32Array::new();

        let mut output = PackedFloat32Array::new();

        for node_weights_index in 0..self.weights.len() {
            let node_weights = self.weights.get(node_weights_index);
            let mut node_output: f32 = 0.;

            for weight_index in 0..node_weights.len() {
                let weight: f32 = node_weights.get(weight_index);

                node_output += self.inputs.get(weight_index) * weight;
            }
            node_output += self.biases.get(node_weights_index);

            output.push(node_output);
        }
        self.outputs = output.clone();
        output
    }

    #[func]
    fn create(in_features_: i8, out_features_: i8) -> Gd<Dense> {
        Gd::from_init_fn(|base| {
            let mut dense_instance = Self {
                in_features: in_features_,
                out_features: out_features_,

                inputs: PackedFloat32Array::new(),
                outputs: PackedFloat32Array::new(),

                weights: array![],
                biases: PackedFloat32Array::new(),

                gradients_w: array![],
                gradients_b: PackedFloat32Array::new(),

                base,
            };
            dense_instance.set_randf_weights_bias_and_zero_gradients();

            dense_instance //return
        })
    }


    #[func]
    fn set_data(&mut self, weights_:Array<PackedFloat32Array>,biases_: PackedFloat32Array) {
        self.weights = weights_;
        self.biases = biases_;
    }

    #[func]
    fn from_data(weights_:Array<PackedFloat32Array>, biases_: PackedFloat32Array) -> Gd<Dense>{
        Gd::from_init_fn(|base| {
            Self {
                in_features: 1,
                out_features: 1,

                inputs: PackedFloat32Array::new(),
                outputs: PackedFloat32Array::new(),

                weights: weights_,
                biases: biases_,

                gradients_w: array![],
                gradients_b: PackedFloat32Array::new(),

                base,
            }
        })
    }

    #[func]
    fn set_randf_weights_bias_and_zero_gradients(&mut self) {
        let mut rng = RandomNumberGenerator::new_gd();
        rng.randomize();

        for out_features_index in 0..self.out_features {
            let mut node_out_weights: PackedFloat32Array = PackedFloat32Array::new();
            let mut row_gradients: PackedFloat32Array = PackedFloat32Array::new();

            for in_feature_index in 0..self.in_features {
                node_out_weights.push(rng.randf_range(-1., 1.));
                row_gradients.push(0.);
            }

            self.weights.push(node_out_weights);
            self.biases.push(rng.randf_range(-1., 1.));

            self.gradients_w.push(row_gradients);
            self.gradients_b.push(0.)
        }
        
    }

    #[func]
    fn calculate_derivative(&self) -> Array<PackedFloat32Array> {
        self.weights.clone()
    }

    #[func]
    fn derivative_respect_inputs(&self) -> Array<PackedFloat32Array> {
        self.weights.clone()
    }

    #[func]
    fn derivative_respect_weights(&self) -> PackedFloat32Array {
        self.inputs.clone()
    }
}
