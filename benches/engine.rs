//! Benchmarks for the hot paths: move application, table build and search.
//!
//! Run with `cargo bench`. Compare two builds with `cargo bench -- --save-baseline a`, then
//! `cargo bench -- --baseline a` after the change.

use criterion::{criterion_group, criterion_main, Criterion};
use rubik::{parse_alg, parse_moveset, Bandage, Kind, Query, Solver, State};
use std::hint::black_box;

const SUNE_X4: &str = "(R U R' U R U2 R')4 (R U R' U')6 (M2 U M2 U2 M2 U M2)";

fn moves(c: &mut Criterion) {
    let alg = parse_alg(SUNE_X4).unwrap();
    c.bench_function("apply_alg", |b| {
        b.iter(|| {
            let mut s = State::solved();
            s.apply_alg(black_box(&alg));
            s
        })
    });
    c.bench_function("pack_unpack", |b| {
        let mut s = State::solved();
        s.apply_alg(&alg);
        b.iter(|| State::unpack(black_box(s.pack())))
    });
}

fn build(c: &mut Criterion) {
    let htm = parse_moveset("htm").unwrap();
    let bandage = Bandage::parse("UF+UFR+UR+U | DF+D").unwrap();
    let mut g = c.benchmark_group("build");
    g.sample_size(10);
    g.bench_function("htm_depth5", |b| {
        b.iter(|| Solver::new(Bandage::none(), htm.clone(), 5, usize::MAX))
    });
    g.bench_function("bandaged_depth7", |b| {
        b.iter(|| Solver::new(bandage.clone(), htm.clone(), 7, usize::MAX))
    });
    g.finish();
}

fn search(c: &mut Criterion) {
    let htm = parse_moveset("htm").unwrap();
    let solver = Solver::new(Bandage::none(), htm, 5, usize::MAX);
    let mut g = c.benchmark_group("search");
    g.sample_size(10);
    g.bench_function("corner3", |b| b.iter(|| solver.search(&Query::new(Kind::Corner3)).unwrap()));
    g.bench_function("edge3_pure", |b| {
        let q = Query { pure: true, ..Query::new(Kind::Edge3) };
        b.iter(|| solver.search(&q).unwrap())
    });
    g.bench_function("corner_twist", |b| {
        b.iter(|| solver.search(&Query::new(Kind::CornerTwist)).unwrap())
    });
    g.finish();
}

criterion_group!(benches, moves, build, search);
criterion_main!(benches);
