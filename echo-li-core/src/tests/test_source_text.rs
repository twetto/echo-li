use std::fs;
use std::path::{Path, PathBuf};

const BAD_PATTERNS: &[&[u32]] = &[
    &[0xfffd],
    &[0x0060, 0x0072, 0x0060, 0x006e],
    &[0x7e5a],
    &[0x5ed8],
    &[0x5cc9],
    &[0x7c21],
    &[0x5e24],
    &[0x5ea5],
    &[0x5f07],
    &[0x7ffb],
    &[0x79ae],
    &[0x649c],
    &[0x875c],
    &[0x876a],
    &[0x64b1],
    &[0x8e4c],
    &[0x8e47],
    &[0x97c1],
    &[0xef85],
    &[0xefc2],
    &[0xe82a],
    &[0xe859],
    &[0x7b28],
    &[0x659f],
];

#[test]
fn source_text_has_no_mojibake_signatures() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core crate has a workspace parent")
        .to_path_buf();
    let mut failures = Vec::new();
    scan_dir(&root, &mut failures);
    assert!(
        failures.is_empty(),
        "mojibake signatures found:\n{}",
        failures.join("\n")
    );
}

fn scan_dir(path: &Path, failures: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == ".git"
            || name == "target"
            || name == ".venv"
            || name == "venv"
            || name == "site-packages"
            || name == "sweep_results"
            || name.starts_with("eqvio_output_")
        {
            continue;
        }
        if path.is_dir() {
            scan_dir(&path, failures);
            continue;
        }
        if !is_checked_source(&path) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for pat in BAD_PATTERNS {
            let pat: String = pat.iter().filter_map(|c| char::from_u32(*c)).collect();
            if let Some(idx) = text.find(&pat) {
                let line = text[..idx].bytes().filter(|b| *b == b'\n').count() + 1;
                failures.push(format!("{}:{line}: {pat:?}", path.display()));
            }
        }
    }
}

fn is_checked_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("rs" | "py" | "toml" | "yaml" | "yml" | "md")
    )
}
