#[cfg(feature = "burn-backend")]
fn main() {
    use burn_import::onnx::ModelGen;
    use std::fs;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    let model_dir = Path::new("assets/models/styles");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let write_empty_glue = |out_dir: &Path| {
        let glue_path = out_dir.join("burn_models.rs");
        let mut f = fs::File::create(&glue_path).unwrap();
        writeln!(f, "// No burn models found at build time").unwrap();
        writeln!(f, "struct BurnModelWrapper;").unwrap();
        writeln!(f, "fn load_burn_models(_device: &<BurnBackend as burn::prelude::Backend>::Device) -> Vec<BurnModelWrapper> {{ Vec::new() }}").unwrap();
        writeln!(f, "fn run_burn_inference(_model: &BurnModelWrapper, input: burn::prelude::Tensor<BurnBackend, 4>) -> burn::prelude::Tensor<BurnBackend, 4> {{ input }}").unwrap();
        writeln!(f, "fn burn_model_names() -> Vec<String> {{ Vec::new() }}").unwrap();
    };

    if !model_dir.exists() {
        println!("cargo:warning=No assets/models/styles/ directory found");
        write_empty_glue(&out_dir);
        return;
    }

    let mut onnx_files: Vec<_> = fs::read_dir(model_dir)
        .expect("Failed to read model directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx")) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    onnx_files.sort();

    if onnx_files.is_empty() {
        println!("cargo:warning=No .onnx files found in assets/models/styles/");
        write_empty_glue(&out_dir);
        return;
    }

    // Generate burn model code from each ONNX file
    let mut model_stems: Vec<String> = Vec::new();

    for onnx_path in &onnx_files {
        let stem = onnx_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Sanitize name for use as Rust identifier
        let rust_name = stem.replace('-', "_").replace('.', "_").to_lowercase();

        println!(
            "cargo:warning=Generating burn model '{}' from: {}",
            rust_name,
            onnx_path.display()
        );

        let out_subdir = format!("burn_models/{}", rust_name);

        // Some ONNX models use ops burn-import can't convert (e.g. dynamic Reshape
        // via Concat of scalar int64s in attention/interpolate layers). Catch panics
        // so the build continues — those models will only be available on ORT backend.
        let input_str = onnx_path.to_str().unwrap().to_string();
        let out_subdir_clone = out_subdir.clone();
        let result = std::panic::catch_unwind(move || {
            ModelGen::new()
                .input(&input_str)
                .out_dir(&out_subdir_clone)
                .embed_states(true)
                .run_from_script();
        });

        match result {
            Ok(()) => model_stems.push(rust_name),
            Err(_) => {
                println!(
                    "cargo:warning=SKIPPED '{}': burn-import cannot convert this model (unsupported ONNX ops). It will only work on ORT backend.",
                    rust_name
                );
                // Clean up partial output
                let partial_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(&out_subdir);
                let _ = fs::remove_dir_all(&partial_dir);
            }
        }
    }

    // Generate the glue module that imports all models and provides dispatch
    let glue_path = out_dir.join("burn_models.rs");
    let mut f = fs::File::create(&glue_path).unwrap();

    // Include each generated model module
    for stem in &model_stems {
        writeln!(f, "#[allow(dead_code, unused_imports, clippy::all)]").unwrap();
        writeln!(f, "mod burn_model_{} {{", stem).unwrap();
        writeln!(
            f,
            "    include!(concat!(env!(\"OUT_DIR\"), \"/burn_models/{}/{}.rs\"));",
            stem, stem
        )
        .unwrap();
        writeln!(f, "}}").unwrap();
        writeln!(f).unwrap();
    }

    // Generate an enum wrapper for all models
    writeln!(f, "#[allow(dead_code)]").unwrap();
    writeln!(f, "enum BurnModelWrapper {{").unwrap();
    for stem in &model_stems {
        writeln!(
            f,
            "    {}(burn_model_{}::Model<BurnBackend>),",
            to_pascal_case(stem),
            stem
        )
        .unwrap();
    }
    writeln!(f, "}}").unwrap();
    writeln!(f).unwrap();

    // Generate load_burn_models function
    writeln!(
        f,
        "#[allow(dead_code)]"
    ).unwrap();
    writeln!(
        f,
        "fn load_burn_models(device: &<BurnBackend as burn::prelude::Backend>::Device) -> Vec<BurnModelWrapper> {{"
    )
    .unwrap();
    writeln!(f, "    vec![").unwrap();
    for stem in &model_stems {
        writeln!(
            f,
            "        BurnModelWrapper::{}(burn_model_{}::Model::from_embedded(device)),",
            to_pascal_case(stem),
            stem
        )
        .unwrap();
    }
    writeln!(f, "    ]").unwrap();
    writeln!(f, "}}").unwrap();
    writeln!(f).unwrap();

    // Generate run_burn_inference function
    writeln!(f, "#[allow(dead_code)]").unwrap();
    writeln!(
        f,
        "fn run_burn_inference(model: &BurnModelWrapper, input: burn::prelude::Tensor<BurnBackend, 4>) -> burn::prelude::Tensor<BurnBackend, 4> {{"
    )
    .unwrap();
    writeln!(f, "    match model {{").unwrap();
    for stem in &model_stems {
        writeln!(
            f,
            "        BurnModelWrapper::{}(m) => m.forward(input),",
            to_pascal_case(stem)
        )
        .unwrap();
    }
    writeln!(f, "    }}").unwrap();
    writeln!(f, "}}").unwrap();
    writeln!(f).unwrap();

    // Generate burn_model_names function (for WASM where filesystem scanning is unavailable)
    writeln!(f, "#[allow(dead_code)]").unwrap();
    writeln!(f, "fn burn_model_names() -> Vec<String> {{").unwrap();
    writeln!(f, "    vec![").unwrap();
    for onnx_path in &onnx_files {
        let name = onnx_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        writeln!(f, "        \"{}\".to_string(),", name).unwrap();
    }
    writeln!(f, "    ]").unwrap();
    writeln!(f, "}}").unwrap();

    // Rerun if any ONNX files change
    println!("cargo:rerun-if-changed=assets/models/styles/");
    for onnx_path in &onnx_files {
        println!("cargo:rerun-if-changed={}", onnx_path.display());
    }
}

#[cfg(feature = "burn-backend")]
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + &chars.collect::<String>(),
            }
        })
        .collect()
}

#[cfg(not(feature = "burn-backend"))]
fn main() {
    // No-op when burn backend is not enabled
}
