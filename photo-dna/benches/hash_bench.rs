use criterion::{Criterion, black_box, criterion_group, criterion_main};
use photo_dna::Hash;

fn hash_benchmark(c: &mut Criterion) {
    // Setup - load test images once
    let img1_path = "tests/image_1.jpg";
    let img2_path = "tests/random.png";
    let img3_path = "tests/image_2.jpg";

    let h1 = Hash::from_image_path(img1_path).unwrap();
    let h2 = Hash::from_image_path(img2_path).unwrap();
    let h3 = Hash::from_image_path(img3_path).unwrap();

    let mut group = c.benchmark_group("Hash Operations");

    // Hash computation
    group.bench_function("from_image", |b| b.iter(|| black_box(Hash::from_image_path(img1_path).unwrap())));

    // Distance metrics
    group.bench_function("distance_euclid/different", |b| b.iter(|| black_box(h1.distance_euclid(&h2))));

    group.bench_function("distance_euclid/similar", |b| b.iter(|| black_box(h1.distance_euclid(&h3))));

    group.bench_function("distance_log2p/different", |b| b.iter(|| black_box(h1.distance_log2p(&h2))));

    group.bench_function("distance_log2p/similar", |b| b.iter(|| black_box(h1.distance_log2p(&h3))));

    // Similarity metrics
    group.bench_function("similarity_euclidian/different", |b| b.iter(|| black_box(h1.similarity_euclidian(&h2))));

    group.bench_function("similarity_euclidian/similar", |b| b.iter(|| black_box(h1.similarity_euclidian(&h3))));

    group.bench_function("similarity_log2p/different", |b| b.iter(|| black_box(h1.similarity_log2p(&h2))));

    group.bench_function("similarity_log2p/similar", |b| b.iter(|| black_box(h1.similarity_log2p(&h3))));

    // Serialization
    group.bench_function("to_hex_string", |b| b.iter(|| h1.to_hex_string()));

    let hex_str = h1.to_hex_string();
    group.bench_function("from_hex_str", |b| b.iter(|| black_box(Hash::from_hex_str(&hex_str)).unwrap()));

    group.finish();
}

criterion_group!(benches, hash_benchmark);
criterion_main!(benches);
