use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zantetsu_core::parser::HeuristicParser;

fn bench_heuristic_parse(c: &mut Criterion) {
    let parser = HeuristicParser::new().unwrap();

    let inputs = vec![
        "[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv",
        "[Erai-raws] Shingeki no Kyojin - The Final Season - 28v2 [1080p][HEVC].mkv",
        "[Judas] Golden Kamuy S3 - 01-12 (1080p) [Batch]",
        "One.Piece.1084.VOSTFR.1080p.WEB.x264-AAC.mkv",
        "[Commie] Steins;Gate - 01 [BD 720p AAC] [5765F5A5].mkv",
    ];

    c.bench_function("heuristic_parse_single", |b| {
        b.iter(|| parser.parse(black_box(inputs[0])).unwrap());
    });

    c.bench_function("heuristic_parse_batch_5", |b| {
        b.iter(|| {
            for input in &inputs {
                let _ = parser.parse(black_box(input)).unwrap();
            }
        });
    });
}

criterion_group!(benches, bench_heuristic_parse);
criterion_main!(benches);
