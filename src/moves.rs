//! Move notation, move tables and algorithm (move sequence) utilities.
//!
//! A *base* is a set of layers turned together (R, M, r, x, ...). A *move* is a base turned
//! 1, 2 or 3 quarter turns in the base's standard direction (R, R2, R').

use crate::geometry::{geom, N_CUBIES, N_ROT};
use std::sync::OnceLock;

pub const N_BASES: usize = 18;
pub const N_MOVES: usize = N_BASES * 3;

/// Names of the bases, 6 per axis: face+, face-, slice, wide+, wide-, whole-cube rotation.
pub const BASE_NAMES: [&str; N_BASES] = [
    "R", "L", "M", "r", "l", "x", //
    "U", "D", "E", "u", "d", "y", //
    "F", "B", "S", "f", "b", "z",
];
/// Layer mask per base kind: bit0 = coordinate -1, bit1 = 0, bit2 = +1.
const KIND_MASK: [u8; 6] = [0b100, 0b001, 0b010, 0b110, 0b011, 0b111];
/// Quarter turns about the +axis for one standard turn of the base (clockwise = -1).
const BASE_DIR: [i32; N_BASES] = [
    -1, 1, 1, -1, 1, -1, // R L M r l x
    -1, 1, 1, -1, 1, -1, // U D E u d y
    -1, 1, -1, -1, 1, -1, // F B S f b z
];

pub type Move = u8;
pub type Alg = Vec<Move>;

#[inline]
pub fn mv(base: usize, amount: u8) -> Move {
    (base * 3 + (amount as usize - 1)) as Move
}
#[inline]
pub fn base_of(m: Move) -> usize {
    m as usize / 3
}
#[inline]
pub fn amount_of(m: Move) -> u8 {
    m % 3 + 1
}
#[inline]
pub fn axis_of_base(b: usize) -> usize {
    b / 6
}
#[inline]
pub fn is_rotation_base(b: usize) -> bool {
    b % 6 == 5
}

pub fn move_name(m: Move) -> String {
    let suffix = match amount_of(m) {
        1 => "",
        2 => "2",
        _ => "'",
    };
    format!("{}{}", BASE_NAMES[base_of(m)], suffix)
}

pub struct MoveTables {
    /// next[m][c][g] = rotation of cubie c after move m when it had rotation g.
    pub next: Vec<[[u8; N_ROT]; N_CUBIES]>,
    /// layer_bit[b][c][g] = (1 << c) if cubie c with rotation g lies in the layers of base b.
    pub layer_bit: Vec<[[u32; N_ROT]; N_CUBIES]>,
}

static TABLES: OnceLock<MoveTables> = OnceLock::new();

pub fn tables() -> &'static MoveTables {
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> MoveTables {
    let geo = geom();
    let mut next = vec![[[0u8; N_ROT]; N_CUBIES]; N_MOVES];
    let mut layer_bit = vec![[[0u32; N_ROT]; N_CUBIES]; N_BASES];
    for b in 0..N_BASES {
        let axis = axis_of_base(b);
        let mask = KIND_MASK[b % 6];
        for c in 0..N_CUBIES {
            for g in 0..N_ROT {
                let p = geo.position(c, g as u8);
                if mask & (1 << (p[axis] + 1)) != 0 {
                    layer_bit[b][c][g] = 1 << c;
                }
            }
        }
        for amount in 1..=3u8 {
            let m = mv(b, amount) as usize;
            let rot = crate::geometry::Mat::quarter(axis, BASE_DIR[b] * amount as i32);
            let r = geo.rots.iter().position(|x| *x == rot).unwrap();
            for c in 0..N_CUBIES {
                for g in 0..N_ROT {
                    next[m][c][g] = if layer_bit[b][c][g] != 0 { geo.mul[r][g] } else { g as u8 };
                }
            }
        }
    }
    MoveTables { next, layer_bit }
}

/// Parse an algorithm. Supports moves (R, U2, F', Rw, r, M, x ...), groups with repetition
/// `(R U R' U')3`, commutators `[R, U]`, conjugates `[R: U]`, and an inverse suffix on groups
/// `(R U)'`. Whitespace is optional.
pub fn parse_alg(s: &str) -> Result<Alg, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    let alg = parse_seq(&chars, &mut pos)?;
    skip_ws(&chars, &mut pos);
    if pos != chars.len() {
        return Err(format!("unexpected '{}' at position {}", chars[pos], pos));
    }
    Ok(alg)
}

fn skip_ws(c: &[char], pos: &mut usize) {
    while *pos < c.len() && (c[*pos].is_whitespace() || c[*pos] == '.') {
        *pos += 1;
    }
}

fn parse_seq(c: &[char], pos: &mut usize) -> Result<Alg, String> {
    let mut out = Vec::new();
    loop {
        skip_ws(c, pos);
        if *pos >= c.len() {
            return Ok(out);
        }
        match c[*pos] {
            ')' | ']' | ',' | ':' => return Ok(out),
            '(' => {
                *pos += 1;
                let inner = parse_seq(c, pos)?;
                expect(c, pos, ')')?;
                out.extend(apply_power(inner, c, pos));
            }
            '[' => {
                *pos += 1;
                let a = parse_seq(c, pos)?;
                skip_ws(c, pos);
                let sep = *c.get(*pos).ok_or("unterminated '['")?;
                if sep != ',' && sep != ':' {
                    return Err(format!("expected ',' or ':' at position {}", pos));
                }
                *pos += 1;
                let b = parse_seq(c, pos)?;
                expect(c, pos, ']')?;
                let mut g = a.clone();
                g.extend(&b);
                g.extend(invert(&a));
                if sep == ',' {
                    g.extend(invert(&b));
                }
                out.extend(apply_power(g, c, pos));
            }
            _ => out.push(parse_move(c, pos)?),
        }
    }
}

fn expect(c: &[char], pos: &mut usize, ch: char) -> Result<(), String> {
    skip_ws(c, pos);
    if c.get(*pos) == Some(&ch) {
        *pos += 1;
        Ok(())
    } else {
        Err(format!("expected '{}' at position {}", ch, pos))
    }
}

fn read_number(c: &[char], pos: &mut usize) -> Option<u32> {
    let start = *pos;
    while *pos < c.len() && c[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos > start { c[start..*pos].iter().collect::<String>().parse().ok() } else { None }
}

fn apply_power(g: Alg, c: &[char], pos: &mut usize) -> Alg {
    let n = read_number(c, pos).unwrap_or(1);
    let g = if c.get(*pos) == Some(&'\'') {
        *pos += 1;
        invert(&g)
    } else {
        g
    };
    let mut out = Vec::with_capacity(g.len() * n as usize);
    for _ in 0..n {
        out.extend(&g);
    }
    out
}

fn parse_move(c: &[char], pos: &mut usize) -> Result<Move, String> {
    let ch = c[*pos];
    let mut base = BASE_NAMES
        .iter()
        .position(|n| n.starts_with(ch))
        .ok_or_else(|| format!("unknown move '{}' at position {}", ch, pos))?;
    *pos += 1;
    if c.get(*pos) == Some(&'w') {
        // Rw == r etc. (only for outer faces)
        if base % 6 > 1 {
            return Err(format!("'w' not allowed after '{}'", ch));
        }
        base += 3;
        *pos += 1;
    }
    let n = read_number(c, pos).unwrap_or(1);
    let mut amount = (n % 4) as i32;
    if c.get(*pos) == Some(&'\'') {
        *pos += 1;
        amount = -amount;
    }
    let amount = amount.rem_euclid(4) as u8;
    if amount == 0 {
        return Err(format!("move with zero turn at position {}", pos));
    }
    Ok(mv(base, amount))
}

pub fn format_alg(alg: &[Move]) -> String {
    alg.iter().map(|&m| move_name(m)).collect::<Vec<_>>().join(" ")
}

pub fn invert(alg: &[Move]) -> Alg {
    alg.iter().rev().map(|&m| mv(base_of(m), 4 - amount_of(m))).collect()
}

/// Cancel and merge moves inside runs of same-axis moves (which commute). Moves on different
/// layer sets are never fused (R M' is not turned into r), so bandage legality is preserved.
pub fn simplify(alg: &[Move]) -> Alg {
    let mut out = Vec::with_capacity(alg.len());
    let mut i = 0;
    while i < alg.len() {
        let axis = axis_of_base(base_of(alg[i]));
        let mut turns = [0u8; 6];
        while i < alg.len() && axis_of_base(base_of(alg[i])) == axis {
            let k = base_of(alg[i]) % 6;
            turns[k] = (turns[k] + amount_of(alg[i])) % 4;
            i += 1;
        }
        for (k, &t) in turns.iter().enumerate() {
            if t != 0 {
                out.push(mv(axis * 6 + k, t));
            }
        }
    }
    // a run that cancelled completely may have joined two runs of the same axis
    if out.len() < alg.len() { simplify(&out) } else { out }
}

/// Length in the slice-turn metric (rotations x/y/z are free).
pub fn length(alg: &[Move]) -> usize {
    alg.iter().filter(|&&m| !is_rotation_base(base_of(m))).count()
}

/// Length in the quarter-turn metric (half turns count 2, rotations are free).
pub fn qtm_length(alg: &[Move]) -> usize {
    alg.iter()
        .filter(|&&m| !is_rotation_base(base_of(m)))
        .map(|&m| if amount_of(m) == 2 { 2 } else { 1 })
        .sum()
}

/// Expand named move sets: "htm"/"faces" (18 face moves), "slices", "wide", or a list of
/// moves/bases such as "R U F M" (a base without suffix adds all three turns).
pub fn parse_moveset(spec: &str) -> Result<Vec<Move>, String> {
    let mut out: Vec<Move> = Vec::new();
    let add_bases = |bases: &[usize], out: &mut Vec<Move>| {
        for &b in bases {
            for a in 1..=3 {
                let m = mv(b, a);
                if !out.contains(&m) {
                    out.push(m);
                }
            }
        }
    };
    for tok in spec.split(|ch: char| ch.is_whitespace() || ch == ',').filter(|t| !t.is_empty()) {
        match tok.to_ascii_lowercase().as_str() {
            "htm" | "faces" => add_bases(&[0, 1, 6, 7, 12, 13], &mut out),
            "slices" => add_bases(&[2, 8, 14], &mut out),
            "wide" => add_bases(&[3, 4, 9, 10, 15, 16], &mut out),
            "stm" => add_bases(&[0, 1, 2, 6, 7, 8, 12, 13, 14], &mut out),
            _ => {
                let chars: Vec<char> = tok.chars().collect();
                let mut p = 0;
                let m = parse_move(&chars, &mut p)?;
                if p != chars.len() {
                    return Err(format!("bad move '{}' in move set", tok));
                }
                let explicit = chars.len() > 1 && !(chars.len() == 2 && chars[1] == 'w');
                if explicit {
                    if !out.contains(&m) {
                        out.push(m);
                    }
                } else {
                    add_bases(&[base_of(m)], &mut out);
                }
            }
        }
    }
    if out.iter().any(|&m| is_rotation_base(base_of(m))) {
        return Err("whole-cube rotations (x, y, z) cannot be used in a search move set".into());
    }
    if out.is_empty() {
        return Err("empty move set".into());
    }
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format() {
        let a = parse_alg("R U2 F' Rw M2 x'").unwrap();
        assert_eq!(format_alg(&a), "R U2 F' r M2 x'");
        assert_eq!(format_alg(&parse_alg("RUR'U'").unwrap()), "R U R' U'");
        assert_eq!(format_alg(&parse_alg("[R, U]").unwrap()), "R U R' U'");
        assert_eq!(format_alg(&parse_alg("[F: R U]").unwrap()), "F R U F'");
        assert_eq!(format_alg(&parse_alg("(R U)2'").unwrap()), "U' R' U' R'");
        assert!(parse_alg("Q").is_err());
    }

    #[test]
    fn simplify_cancels() {
        let a = parse_alg("R L R' U U F F' U2").unwrap();
        assert_eq!(format_alg(&simplify(&a)), "L");
        let b = parse_alg("R r R").unwrap();
        assert_eq!(format_alg(&simplify(&b)), "R2 r");
    }

    #[test]
    fn movesets() {
        assert_eq!(parse_moveset("htm").unwrap().len(), 18);
        assert_eq!(parse_moveset("R U2 F").unwrap().len(), 7);
        assert!(parse_moveset("x").is_err());
    }
}
