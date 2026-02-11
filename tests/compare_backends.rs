//! Compare ort and burn inference outputs for the same model and input.
//! Run with: cargo test --test compare_backends --features "ort-backend,burn-backend" -- --nocapture

#[cfg(all(feature = "ort-backend", feature = "burn-backend"))]
mod tests {
    use ndarray::Array4;

    fn create_test_input(w: usize, h: usize) -> Vec<u8> {
        let mut pixels = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                pixels[idx] = ((x as f32 / w as f32) * 255.0) as u8; // R = gradient
                pixels[idx + 1] = ((y as f32 / h as f32) * 255.0) as u8; // G = gradient
                pixels[idx + 2] = 128; // B = constant
                pixels[idx + 3] = 255; // A = opaque
            }
        }
        pixels
    }

    fn ort_inference(pixels: &[u8], w: usize, h: usize) -> Vec<f32> {
        use ort::session::Session;
        use ort::value::Tensor;

        let mut input = Array4::<f32>::zeros((1, 3, h, w));
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                input[[0, 0, y, x]] = pixels[idx] as f32 / 255.0;
                input[[0, 1, y, x]] = pixels[idx + 1] as f32 / 255.0;
                input[[0, 2, y, x]] = pixels[idx + 2] as f32 / 255.0;
            }
        }

        let mut session = Session::builder()
            .unwrap()
            .with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])
            .unwrap()
            .commit_from_file("assets/models/styles/test_style_fixed.onnx")
            .expect("Failed to load ONNX model");

        let input_tensor = Tensor::from_array(input).unwrap();
        let outputs = session
            .run(ort::inputs!["input" => input_tensor])
            .expect("Inference failed");
        let output = outputs["output"]
            .try_extract_array::<f32>()
            .expect("Failed to extract output");

        output.as_slice().unwrap().to_vec()
    }

    fn burn_inference(pixels: &[u8], w: usize, h: usize) -> Vec<f32> {
        use burn::backend::NdArray;
        use burn::prelude::*;

        type B = NdArray<f32>;

        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        // Load the burn model
        // Use the same module that build.rs generates
        #[allow(dead_code, unused_imports, clippy::all)]
        mod burn_model {
            include!(concat!(
                env!("OUT_DIR"),
                "/burn_models/test_style_fixed/test_style_fixed.rs"
            ));
        }

        let model: burn_model::Model<B> = burn_model::Model::default();

        // Convert pixels to tensor
        let mut data = vec![0.0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                data[0 * h * w + y * w + x] = pixels[idx] as f32 / 255.0;
                data[1 * h * w + y * w + x] = pixels[idx + 1] as f32 / 255.0;
                data[2 * h * w + y * w + x] = pixels[idx + 2] as f32 / 255.0;
            }
        }
        let input_tensor = Tensor::<B, 4>::from_floats(
            burn::tensor::TensorData::new(data, [1, 3, h, w]),
            &device,
        );

        let output_tensor = model.forward(input_tensor);
        let output_data: Vec<f32> = output_tensor.into_data().to_vec().unwrap();
        output_data
    }

    #[test]
    fn compare_ort_and_burn_inference() {
        let w = 512;
        let h = 288;
        let pixels = create_test_input(w, h);

        println!("Running ort inference...");
        let ort_output = ort_inference(&pixels, w, h);

        println!("Running burn inference...");
        let burn_output = burn_inference(&pixels, w, h);

        assert_eq!(
            ort_output.len(),
            burn_output.len(),
            "Output sizes differ: ort={}, burn={}",
            ort_output.len(),
            burn_output.len()
        );

        // Compare outputs
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f64;
        for (i, (a, b)) in ort_output.iter().zip(burn_output.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            total_diff += diff as f64;

            if diff > 0.01 && i < 100 {
                println!(
                    "  [{}] ort={:.6}, burn={:.6}, diff={:.6}",
                    i, a, b, diff
                );
            }
        }
        let mean_diff = total_diff / ort_output.len() as f64;

        println!("\n=== Comparison Results ===");
        println!("Output size: {}", ort_output.len());
        println!("Max diff:  {:.6}", max_diff);
        println!("Mean diff: {:.8}", mean_diff);
        println!(
            "ORT range:  [{:.4}, {:.4}]",
            ort_output.iter().cloned().reduce(f32::min).unwrap(),
            ort_output.iter().cloned().reduce(f32::max).unwrap()
        );
        println!(
            "Burn range: [{:.4}, {:.4}]",
            burn_output.iter().cloned().reduce(f32::min).unwrap(),
            burn_output.iter().cloned().reduce(f32::max).unwrap()
        );

        // Allow some tolerance for floating point differences
        assert!(
            max_diff < 0.01,
            "Max diff between ort and burn outputs is too large: {} (should be < 0.01)",
            max_diff
        );
    }
}
