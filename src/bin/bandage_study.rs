//! Difficulty study: profile N random bandaged cubes plus the plain 3x3x3 baseline.
//! Degenerate puzzles (a single face is all that ever turns) are skipped.
//!
//! ```text
//! cargo run --release --bin bandage_study -- --n 100 --seed 1 --out study
//! ```
//!
//! Options: `--n`, `--seed`, `--bonds MIN-MAX` (glued pairs per puzzle, default 1-12),
//! `--max-nodes` (table size, default 10M), `--samples`, `--ball-nodes`, `--alg-time` (seconds),
//! `--out` (directory for results.csv / results.json), `--bandage SPEC` (profile one puzzle and exit),
//! `--possible SPEC` (print how many edge 3-cycles, corner 3-cycles, flips and twists are possible).

use rubik::study::{
    profile, random_bandage, Profile, Rng, StudyConfig, ALG_KINDS, CUBE333_GODS_NUMBER, CUBE333_MEAN_DISTANCE,
    CUBE333_STATES,
};
use rubik::Bandage;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

struct Args {
    n: usize,
    bonds: (usize, usize),
    out: PathBuf,
    bandage: Option<String>,
    possible: Option<String>,
    cfg: StudyConfig,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { n: 100, bonds: (1, 12), out: PathBuf::from("study"), bandage: None, possible: None, cfg: StudyConfig::default() };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let val = argv.get(i + 1).ok_or_else(|| format!("missing value for {}", argv[i]))?;
        let num = || val.replace('_', "").parse::<usize>().map_err(|e| format!("{}: {}", argv[i], e));
        match argv[i].as_str() {
            "--n" => a.n = num()?,
            "--seed" => a.cfg.seed = num()? as u64,
            "--max-nodes" => a.cfg.max_nodes = num()?,
            "--samples" => a.cfg.samples = num()?,
            "--ball-nodes" => a.cfg.ball_nodes = num()?,
            "--alg-time" => a.cfg.alg_time_limit = val.parse().map_err(|e| format!("--alg-time: {}", e))?,
            "--out" => a.out = PathBuf::from(val),
            "--bandage" => a.bandage = Some(val.clone()),
            "--possible" => a.possible = Some(val.clone()),
            "--bonds" => {
                let (lo, hi) = val.split_once('-').ok_or("--bonds expects MIN-MAX")?;
                a.bonds = (lo.parse().map_err(|_| "bad --bonds")?, hi.parse().map_err(|_| "bad --bonds")?);
            }
            other => return Err(format!("unknown option {}", other)),
        }
        i += 2;
    }
    Ok(a)
}

fn opt<T: ToString>(x: Option<T>) -> String {
    x.map(|v| v.to_string()).unwrap_or_default()
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn json_opt<T: ToString>(x: Option<T>) -> String {
    x.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

const CSV_HEADER: &str = "id,bonds,spec,blocks,bandaged_cubies,free_corners,free_edges,legal_at_solved,complete,\
depth,table_states,states,states_exact,shapes,group_order,log10_states,gods_number,gods_lower,gods_counting,mean_distance,\
mean_distance_exact,samples,samples_resolved,branching,antipode_distance,\
edge3,edge3_exact,edge3_possible,edge3_cov,corner3,corner3_exact,corner3_possible,corner3_cov,\
edge_flip,edge_flip_exact,edge_flip_possible,edge_flip_cov,\
corner_twist,corner_twist_exact,corner_twist_possible,corner_twist_cov,seconds";

fn csv_row(id: &str, bonds: usize, p: &Profile) -> String {
    let mut s = format!(
        "{},{},\"{}\",{},{},{},{},{},{},{},{},{:.6e},{},{},{},{:.4},{},{},{},{:.4},{},{},{},{:.4},{}",
        id,
        bonds,
        p.spec,
        p.blocks,
        p.bandaged_cubies,
        p.free_corners,
        p.free_edges,
        p.legal_at_solved,
        p.complete,
        p.depth,
        p.table_states,
        p.states,
        p.states_exact,
        p.shapes,
        p.group_order,
        p.states.log10(),
        opt(p.gods_number),
        p.gods_lower,
        opt(p.gods_counting),
        p.mean_distance,
        p.mean_distance_exact,
        p.samples,
        p.samples_resolved,
        p.branching,
        p.antipode_distance
    );
    for a in &p.algs {
        let _ = write!(s, ",{},{},{},{:.3}", opt(a.min_len), a.exact, a.possible, a.coverage);
    }
    let _ = write!(s, ",{:.2}", p.seconds);
    s
}

fn json_obj(id: &str, bonds: usize, p: &Profile) -> String {
    let algs: Vec<String> = ALG_KINDS
        .iter()
        .zip(&p.algs)
        .map(|(k, a)| {
            format!(
                "{}: {{\"min_len\": {}, \"exact\": {}, \"searched_to\": {}, \"possible\": {}, \"coverage\": {:.4}, \"example\": {}}}",
                json_str(k.name()),
                json_opt(a.min_len),
                a.exact,
                a.searched_to,
                a.possible,
                a.coverage,
                json_str(&a.example)
            )
        })
        .collect();
    format!(
        "{{\"id\": {}, \"bonds\": {}, \"spec\": {}, \"blocks\": {}, \"bandaged_cubies\": {}, \"free_corners\": {}, \
\"free_edges\": {}, \"legal_at_solved\": {}, \"complete\": {}, \"depth\": {}, \"layer_sizes\": {:?}, \
\"table_states\": {}, \"states\": {:e}, \"states_exact\": {}, \"shapes\": {}, \"group_order\": \"{}\", \"gods_number\": {}, \"gods_lower\": {}, \
\"gods_counting\": {}, \"mean_distance\": {:.4}, \"mean_distance_exact\": {}, \"samples\": {}, \
\"samples_resolved\": {}, \"branching\": {:.4}, \"antipode_distance\": {}, \"antipode\": {}, \"algs\": {{{}}}, \
\"seconds\": {:.2}}}",
        json_str(id),
        bonds,
        json_str(&p.spec),
        p.blocks,
        p.bandaged_cubies,
        p.free_corners,
        p.free_edges,
        p.legal_at_solved,
        p.complete,
        p.depth,
        p.layer_sizes,
        p.table_states,
        p.states,
        p.states_exact,
        p.shapes,
        p.group_order,
        json_opt(p.gods_number),
        p.gods_lower,
        json_opt(p.gods_counting),
        p.mean_distance,
        p.mean_distance_exact,
        p.samples,
        p.samples_resolved,
        p.branching,
        p.antipode_distance,
        json_str(&p.antipode),
        algs.join(", "),
        p.seconds
    )
}

fn summary(id: &str, p: &Profile) -> String {
    let god = match (p.gods_number, p.gods_counting) {
        (Some(g), _) => format!("{}", g),
        (None, Some(c)) => format!("~{} (>={})", c, p.gods_lower),
        _ => format!(">={}", p.gods_lower),
    };
    let algs: Vec<String> = p
        .algs
        .iter()
        .map(|a| match a.min_len {
            Some(l) => format!("{}{}", l, if a.exact { "" } else { "?" }),
            None => "-".into(),
        })
        .collect();
    format!(
        "{:>5} {:>8} log10N={:>5.2}{} god={:<11} mean={:>5.2}{} branch={:>5.2} algs[e3 c3 ef ct]={} ({:.1}s)  {}",
        id,
        if p.complete { "complete" } else { "sampled" },
        p.states.log10(),
        if p.states_exact { " " } else { "+" },
        god,
        p.mean_distance,
        if p.mean_distance_exact { " " } else { "+" },
        p.branching,
        algs.join(" "),
        p.seconds,
        p.spec
    )
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(2);
        }
    };
    if let Some(spec) = &args.possible {
        let b = Bandage::parse(spec).unwrap_or_else(|e| panic!("bad bandage: {}", e));
        let counts = rubik::study::effect_counts(&b, args.cfg.max_shapes).expect("too many shapes");
        println!("{}", counts.map(|c| c.0.to_string()).join(" "));
        return;
    }
    if let Some(spec) = &args.bandage {
        let b = Bandage::parse(spec).unwrap_or_else(|e| panic!("bad bandage: {}", e));
        let p = profile(&b, &args.cfg);
        println!("{}", summary("one", &p));
        println!("{}", json_obj("one", 0, &p));
        return;
    }

    fs::create_dir_all(&args.out).expect("create output directory");
    let mut csv = vec![CSV_HEADER.to_string()];
    let mut json = Vec::new();

    eprintln!("baseline: unbandaged 3x3x3 (state count computed, God's number and mean from the literature)");
    let mut base = profile(&Bandage::none(), &args.cfg);
    // The 3x3x3 is far too deep for the sampler (it only gets lower bounds): use the known values.
    assert_eq!(base.states, CUBE333_STATES);
    base.gods_number = Some(CUBE333_GODS_NUMBER);
    base.gods_lower = CUBE333_GODS_NUMBER;
    base.mean_distance = CUBE333_MEAN_DISTANCE;
    base.mean_distance_exact = true;
    eprintln!("{}", summary("333", &base));
    csv.push(csv_row("333", 0, &base));
    json.push(json_obj("333", 0, &base));

    let mut rng = Rng::new(args.cfg.seed);
    let mut seen = HashSet::new();
    let mut id = 0;
    let mut rejected = 0;
    while id < args.n {
        let bonds = args.bonds.0 + rng.below(args.bonds.1 - args.bonds.0 + 1);
        let b = random_bandage(&mut rng, bonds);
        let p0 = profile_cheap_check(&b);
        // skip duplicates and puzzles that cannot turn at all
        if !p0 || !seen.insert(b.spec()) {
            continue;
        }
        let p = profile(&b, &args.cfg);
        if p.states <= 4.0 {
            rejected += 1; // degenerate: a single face is the only thing that ever turns
            continue;
        }
        id += 1;
        let name = format!("B{:03}", id);
        eprintln!("{}", summary(&name, &p));
        csv.push(csv_row(&name, bonds, &p));
        json.push(json_obj(&name, bonds, &p));
        // write as we go so a long run can be inspected or interrupted
        fs::write(args.out.join("results.csv"), csv.join("\n") + "\n").expect("write csv");
        fs::write(args.out.join("results.json"), format!("[\n{}\n]\n", json.join(",\n"))).expect("write json");
    }
    fs::write(args.out.join("results.csv"), csv.join("\n") + "\n").expect("write csv");
    fs::write(args.out.join("results.json"), format!("[\n{}\n]\n", json.join(",\n"))).expect("write json");
    eprintln!("wrote {} ({} degenerate puzzles skipped)", args.out.display(), rejected);
}

/// At least one face can turn in the solved state.
fn profile_cheap_check(b: &Bandage) -> bool {
    let moves = rubik::parse_moveset("htm").unwrap();
    rubik::study::legal_face_turns(b, &rubik::State::solved(), &moves) > 0
}
