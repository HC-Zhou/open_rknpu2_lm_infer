use std::process::Command;

#[test]
fn openmp_wait_defaults_and_overrides_are_applied_at_startup() {
    for (policy, spins, expected_policy, expected_spins) in [
        (None, None, "PASSIVE", "0"),
        (Some("PASSIVE"), None, "PASSIVE", "0"),
        (None, Some("1234"), "PASSIVE", "1234"),
        (Some("ACTIVE"), Some("1234"), "ACTIVE", "1234"),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_open_rknpu2_lm_infer"));
        command
            .arg("--help")
            .env_remove("OMP_WAIT_POLICY")
            .env_remove("GOMP_SPINCOUNT")
            .env("OMP_DISPLAY_ENV", "VERBOSE");
        if let Some(policy) = policy {
            command.env("OMP_WAIT_POLICY", policy);
        }
        if let Some(spins) = spins {
            command.env("GOMP_SPINCOUNT", spins);
        }
        let output = command.output().unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{stderr}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Usage:"), "{stdout}");
        // Check libgomp's parsed state in the final process, not just getenv.
        let display = stderr
            .rsplit_once("OPENMP DISPLAY ENVIRONMENT BEGIN")
            .expect("libgomp environment report")
            .1;
        assert!(
            display.contains(&format!("OMP_WAIT_POLICY = '{expected_policy}'")),
            "{stderr}"
        );
        assert!(
            display.contains(&format!("GOMP_SPINCOUNT = '{expected_spins}'")),
            "{stderr}"
        );
    }
}
