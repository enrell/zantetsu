use zantetsu_trainer::run_training;

fn main() {
    if let Err(e) = run_training() {
        eprintln!("Training failed: {}", e);
        std::process::exit(1);
    }
}
