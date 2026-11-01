//! The 4x4x4 cube: geometry, move notation, state, bandaging and effects.
//!
//! Coordinates are doubled so every cubie sits on integer points: each coordinate is one of
//! -3, -1, 1, 3 (+x = R, +y = U, +z = F, as on the 3x3x3). The 56 cubies are 8 corners,
//! 24 wings (edge pieces) and 24 centers. As on the 3x3x3 a cubie's state is one of the 24 cube
//! rotations. A wing or a center has a different position under each rotation, so its position
//! fixes its orientation (wings cannot be flipped in place).
//!
//! Cubie names have one letter per axis, uppercase for an outer coordinate (±3) and lowercase for
//! an inner one (±1), in any order: corners `UFR`, wings `UFr` (on the UF edge, next to R),
//! centers `Ufr` (on U, in the quadrant towards F and R).
//!
//! The four centers of a colour look the same. Unless glued in a block, centers are compared by
//! colour only: [`Bandage4::canonical`] relabels them, which gives the packed form and effects.
//!
//! Notation (per axis R/U/F): `R` `L` outer layers, `2R` `2L` inner slices, `r` = `Rw` the two
//! right layers, `l` = `Lw`, `x` whole-cube rotation; suffixes `2` and `'` as on the 3x3x3.

use crate::cube::Effect;
use crate::geometry::{face_of_normal, geom, Vec3, FACES, FACE_NORMALS, ID_ROT, NAMES, N_ROT};
use crate::moves::{amount_of, base_of, mv, parse_alg_with, parse_turns, simplify_with, Alg, Move};
use crate::search::{Kind, PackedKey, Puzzle, Query, Reject};
use std::hash::{Hash, Hasher};
use std::ops::BitAnd;
use std::sync::OnceLock;

pub const N_CUBIES4: usize = 56;
pub const WING0: usize = 8;
pub const CENTER0_4: usize = 32;
pub const N_FACELETS4: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Piece4 {
    Corner,
    Wing,
    Center,
}

pub fn piece4(c: usize) -> Piece4 {
    if c < WING0 {
        Piece4::Corner
    } else if c < CENTER0_4 {
        Piece4::Wing
    } else {
        Piece4::Center
    }
}

/// Doubled coordinate of layer `k` (0..4, counted from the negative side).
fn coord(k: usize) -> i8 {
    2 * k as i8 - 3
}

/// Position of the cubie and outward normal of facelet `i` (0..96, URFDLB, each face row by
/// row in the same layout as the 3x3x3's facelets).
pub fn facelet_geometry4(i: usize) -> (Vec3, Vec3) {
    let (f, r, c) = (i / 16, coord(i % 16 / 4), coord(i % 4));
    let p = match f {
        0 => [c, 3, r],
        1 => [3, -r, -c],
        2 => [c, -r, 3],
        3 => [c, -3, -r],
        4 => [-3, -r, c],
        _ => [-c, -r, -3],
    };
    (p, FACE_NORMALS[f])
}

/// Letter of the face in direction `sign` along `axis`, lowercase for an inner coordinate.
fn letter(axis: usize, v: i8) -> char {
    let mut n = [0i8; 3];
    n[axis] = v.signum();
    let ch = FACES[face_of_normal(n)];
    if v.abs() == 3 { ch } else { ch.to_ascii_lowercase() }
}

/// Precomputed geometry of the 4x4x4. Built once on first use.
pub struct Geo4 {
    pub home: [Vec3; N_CUBIES4],
    pub names: Vec<&'static str>,
    /// slot[c][g]: index of the cubie whose home is where cubie `c` sits under rotation g.
    pub slot: [[u8; N_ROT]; N_CUBIES4],
    /// ori[c][g]: corner orientation as on the 3x3x3; 0 for wings and centers.
    pub ori: [[u8; N_ROT]; N_CUBIES4],
    /// unpack[c][slot][ori] -> rotation.
    pub unpack: [[[u8; 3]; N_CUBIES4]; N_CUBIES4],
    /// Cubie whose home shows facelet i.
    pub facelet_slot: [u8; N_FACELETS4],
}

static GEO4: OnceLock<Geo4> = OnceLock::new();

pub fn geo4() -> &'static Geo4 {
    GEO4.get_or_init(Geo4::build)
}

impl Geo4 {
    fn build() -> Geo4 {
        let g3 = geom();
        let mut home: Vec<Vec3> = Vec::with_capacity(N_CUBIES4);
        let mut names: Vec<String> = Vec::with_capacity(N_CUBIES4);
        // corners: the 3x3x3's, scaled
        for c in 0..8 {
            home.push(g3.home[c].map(|v| 3 * v));
            names.push(NAMES[c].to_string());
        }
        // wings: two per 3x3x3 edge, on either side of its midpoint
        for e in 8..20 {
            let p = g3.home[e].map(|v| 3 * v);
            let axis = p.iter().position(|&v| v == 0).unwrap();
            for d in [1, -1] {
                let mut q = p;
                q[axis] = d;
                home.push(q);
                names.push(format!("{}{}", NAMES[e], letter(axis, d)));
            }
        }
        // centers: in facelet order, face by face
        for i in 0..N_FACELETS4 {
            let (p, n) = facelet_geometry4(i);
            if p.iter().filter(|v| v.abs() == 3).count() == 1 {
                let face = n.iter().position(|&v| v != 0).unwrap();
                let mut name = FACES[i / 16].to_string();
                for axis in [1, 2, 0] {
                    if axis != face {
                        name.push(letter(axis, p[axis]));
                    }
                }
                home.push(p);
                names.push(name);
            }
        }
        assert_eq!(home.len(), N_CUBIES4);
        let mut geo = Geo4 {
            home: home.try_into().unwrap(),
            names: names.into_iter().map(|s| &*Box::leak(s.into_boxed_str())).collect(),
            slot: [[0; N_ROT]; N_CUBIES4],
            ori: [[0; N_ROT]; N_CUBIES4],
            unpack: [[[0; 3]; N_CUBIES4]; N_CUBIES4],
            facelet_slot: [0; N_FACELETS4],
        };
        let at = |home: &[Vec3; N_CUBIES4], p: Vec3| home.iter().position(|&h| h == p).unwrap();
        for c in 0..N_CUBIES4 {
            for g in 0..N_ROT {
                if c < WING0 {
                    geo.slot[c][g] = g3.slot[c][g];
                    geo.ori[c][g] = g3.ori[c][g];
                } else {
                    let s = at(&geo.home, g3.apply(g as u8, geo.home[c]));
                    geo.slot[c][g] = s as u8;
                    geo.unpack[c][s][0] = g as u8;
                }
            }
            if c < WING0 {
                for s in 0..WING0 {
                    geo.unpack[c][s] = g3.unpack[c][s];
                }
            }
        }
        for i in 0..N_FACELETS4 {
            geo.facelet_slot[i] = at(&geo.home, facelet_geometry4(i).0) as u8;
        }
        geo
    }
}

/// Parse a cubie name: one letter per axis, uppercase outer, lowercase inner ("UFR", "rUF", "Ufr").
pub fn parse_cubie4(name: &str) -> Option<usize> {
    let mut p = [0i8; 3];
    let name = name.trim();
    if name.chars().count() != 3 {
        return None;
    }
    for ch in name.chars() {
        let f = FACES.iter().position(|&f| f == ch.to_ascii_uppercase())?;
        let n = FACE_NORMALS[f];
        let axis = n.iter().position(|&v| v != 0).unwrap();
        if p[axis] != 0 {
            return None;
        }
        p[axis] = n[axis] * if ch.is_ascii_uppercase() { 3 } else { 1 };
    }
    geo4().home.iter().position(|&h| h == p)
}

// ---- moves ----

pub const N_BASES4: usize = 21;
const PER_AXIS: usize = 7;

/// Names of the bases, 7 per axis: outer+, outer-, inner+, inner-, wide+, wide-, rotation.
pub const BASE_NAMES4: [&str; N_BASES4] = [
    "R", "L", "2R", "2L", "r", "l", "x", //
    "U", "D", "2U", "2D", "u", "d", "y", //
    "F", "B", "2F", "2B", "f", "b", "z",
];
/// Layers of each base kind: bit k = the layer at coordinate `coord(k)`.
const LAYERS4: [u8; PER_AXIS] = [0b1000, 0b0001, 0b0100, 0b0010, 0b1100, 0b0011, 0b1111];
/// Quarter turns about the +axis for one standard turn (clockwise seen from the + side = -1).
const DIR4: [i32; PER_AXIS] = [-1, 1, -1, 1, -1, 1, -1];

pub fn is_rotation_base4(b: usize) -> bool {
    b % PER_AXIS == PER_AXIS - 1
}

pub fn move_name4(m: Move) -> String {
    let suffix = match amount_of(m) {
        1 => "",
        2 => "2",
        _ => "'",
    };
    format!("{}{}", BASE_NAMES4[base_of(m)], suffix)
}

pub struct MoveTables4 {
    /// next[m][c][g] = rotation of cubie c after move m when it had rotation g.
    pub next: Vec<[[u8; N_ROT]; N_CUBIES4]>,
    /// layer_bit[b][c][g] = (1 << c) if cubie c with rotation g lies in the layers of base b.
    pub layer_bit: Vec<[[u64; N_ROT]; N_CUBIES4]>,
}

static TABLES4: OnceLock<MoveTables4> = OnceLock::new();

pub fn tables4() -> &'static MoveTables4 {
    TABLES4.get_or_init(|| {
        let g3 = geom();
        let geo = geo4();
        let mut next = vec![[[0u8; N_ROT]; N_CUBIES4]; N_BASES4 * 3];
        let mut layer_bit = vec![[[0u64; N_ROT]; N_CUBIES4]; N_BASES4];
        for b in 0..N_BASES4 {
            let (axis, kind) = (b / PER_AXIS, b % PER_AXIS);
            for c in 0..N_CUBIES4 {
                for g in 0..N_ROT {
                    let p = g3.apply(g as u8, geo.home[c]);
                    if LAYERS4[kind] & (1 << ((p[axis] + 3) / 2)) != 0 {
                        layer_bit[b][c][g] = 1 << c;
                    }
                }
            }
            for amount in 1..=3u8 {
                let m = mv(b, amount) as usize;
                let rot = crate::geometry::Mat::quarter(axis, DIR4[kind] * amount as i32);
                let r = g3.rots.iter().position(|x| *x == rot).unwrap();
                for c in 0..N_CUBIES4 {
                    for g in 0..N_ROT {
                        next[m][c][g] = if layer_bit[b][c][g] != 0 { g3.mul[r][g] } else { g as u8 };
                    }
                }
            }
        }
        MoveTables4 { next, layer_bit }
    })
}

/// One 4x4x4 move token: `R`, `2R`, `Rw` / `r`, `x`, with a turn suffix.
fn parse_move4(c: &[char], pos: &mut usize) -> Result<Move, String> {
    let at = *pos;
    let inner = c[at] == '2' && c.get(at + 1).is_some_and(|ch| "RLUDFB".contains(*ch));
    if inner {
        *pos += 1;
    }
    let ch = c[*pos];
    let face = "RLUDFB".find(ch.to_ascii_uppercase());
    let base = match (face, ch) {
        (_, 'x') => 6,
        (_, 'y') => PER_AXIS + 6,
        (_, 'z') => 2 * PER_AXIS + 6,
        (Some(f), _) => {
            let (axis, side) = (f / 2, f % 2);
            let wide = ch.is_ascii_lowercase() || c.get(*pos + 1) == Some(&'w');
            if c.get(*pos + 1) == Some(&'w') {
                *pos += 1;
            }
            if inner && wide {
                return Err(format!("unsupported move at position {}", at));
            }
            axis * PER_AXIS + side + if inner { 2 } else if wide { 4 } else { 0 }
        }
        _ => return Err(format!("unknown 4x4x4 move '{}' at position {}", ch, at)),
    };
    *pos += 1;
    parse_turns(c, pos, base)
}

/// Parse a 4x4x4 algorithm (same grouping syntax as [`crate::moves::parse_alg`]).
pub fn parse_alg4(s: &str) -> Result<Alg, String> {
    parse_alg_with(s, parse_move4)
}

pub fn format_alg4(alg: &[Move]) -> String {
    alg.iter().map(|&m| move_name4(m)).collect::<Vec<_>>().join(" ")
}

pub fn simplify4(alg: &[Move]) -> Alg {
    simplify_with(alg, PER_AXIS)
}

/// Length in the slice-turn metric (every layer or block of layers counts 1, rotations are free).
pub fn length4(alg: &[Move]) -> usize {
    alg.iter().filter(|&&m| !is_rotation_base4(base_of(m))).count()
}

/// Length in the quarter-turn metric (half turns count 2, rotations are free).
pub fn qtm_length4(alg: &[Move]) -> usize {
    alg.iter().filter(|&&m| !is_rotation_base4(base_of(m))).map(|&m| if amount_of(m) == 2 { 2 } else { 1 }).sum()
}

/// Move sets: "htm"/"faces" (18 outer turns), "inner"/"slices" (inner slices), "wide",
/// "stm" (every single layer: outer + inner), or a list of moves/bases such as "R 2R U".
pub fn parse_moveset4(spec: &str) -> Result<Vec<Move>, String> {
    let mut out: Vec<Move> = Vec::new();
    let add_kinds = |kinds: &[usize], out: &mut Vec<Move>| {
        for axis in 0..3 {
            for &k in kinds {
                for a in 1..=3 {
                    let m = mv(axis * PER_AXIS + k, a);
                    if !out.contains(&m) {
                        out.push(m);
                    }
                }
            }
        }
    };
    for tok in spec.split(|ch: char| ch.is_whitespace() || ch == ',').filter(|t| !t.is_empty()) {
        match tok.to_ascii_lowercase().as_str() {
            "htm" | "faces" => add_kinds(&[0, 1], &mut out),
            "inner" | "slices" => add_kinds(&[2, 3], &mut out),
            "wide" => add_kinds(&[4, 5], &mut out),
            "stm" => add_kinds(&[0, 1, 2, 3], &mut out),
            _ => {
                let chars: Vec<char> = tok.chars().collect();
                let mut p = 0;
                let m = parse_move4(&chars, &mut p)?;
                if p != chars.len() {
                    return Err(format!("bad move '{}' in move set", tok));
                }
                let explicit = matches!(chars.last(), Some(ch) if ch.is_ascii_digit() || *ch == '\'');
                let ms: Vec<Move> = if explicit { vec![m] } else { (1..=3).map(|a| mv(base_of(m), a)).collect() };
                for m in ms {
                    if !out.contains(&m) {
                        out.push(m);
                    }
                }
            }
        }
    }
    if out.iter().any(|&m| is_rotation_base4(base_of(m))) {
        return Err("whole-cube rotations (x, y, z) cannot be used in a search move set".into());
    }
    if out.is_empty() {
        return Err("empty move set".into());
    }
    out.sort();
    Ok(out)
}

// ---- state ----

/// Rotation of every cubie relative to its home.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State4 {
    pub rot: [u8; N_CUBIES4],
}

impl Default for State4 {
    fn default() -> Self {
        State4 { rot: [ID_ROT; N_CUBIES4] }
    }
}

impl State4 {
    pub fn solved() -> State4 {
        State4::default()
    }

    #[inline]
    pub fn apply_move(&mut self, m: Move) {
        let next = &tables4().next[m as usize];
        for c in 0..N_CUBIES4 {
            self.rot[c] = next[c][self.rot[c] as usize];
        }
    }

    pub fn apply_alg(&mut self, alg: &[Move]) {
        for &m in alg {
            self.apply_move(m);
        }
    }

    /// Bitmask of the cubies currently inside the layers of base `b`.
    #[inline]
    pub fn layer_mask(&self, b: usize) -> u64 {
        let lb = &tables4().layer_bit[b];
        let mut mask = 0;
        for c in 0..N_CUBIES4 {
            mask |= lb[c][self.rot[c] as usize];
        }
        mask
    }

    pub fn slot(&self, c: usize) -> usize {
        geo4().slot[c][self.rot[c] as usize] as usize
    }

    pub fn ori(&self, c: usize) -> u8 {
        geo4().ori[c][self.rot[c] as usize]
    }

    /// Cubie occupying each slot.
    pub fn occupants(&self) -> [usize; N_CUBIES4] {
        let mut occ = [0; N_CUBIES4];
        for c in 0..N_CUBIES4 {
            occ[self.slot(c)] = c;
        }
        occ
    }

    /// 96-character facelet string in URFDLB order (each face row by row).
    pub fn facelets(&self) -> String {
        let occ = self.occupants();
        let (g3, geo) = (geom(), geo4());
        (0..N_FACELETS4)
            .map(|i| {
                let c = occ[geo.facelet_slot[i] as usize];
                let n = facelet_geometry4(i).1;
                FACES[face_of_normal(g3.apply(g3.inv[self.rot[c] as usize], n))]
            })
            .collect()
    }
}

/// Cubie currently showing each of the 96 facelets.
pub fn facelet_cubies4(s: &State4) -> [usize; N_FACELETS4] {
    let occ = s.occupants();
    let geo = geo4();
    std::array::from_fn(|i| occ[geo.facelet_slot[i] as usize])
}

// ---- packing: 5 bits per cubie, 12 cubies per word ----

/// A packed 4x4x4 state (canonical: free centers are labelled by colour).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key4(pub [u64; 5]);

impl Hash for Key4 {
    fn hash<H: Hasher>(&self, h: &mut H) {
        for w in self.0 {
            h.write_u64(w);
        }
    }
}

impl BitAnd for Key4 {
    type Output = Key4;
    #[inline]
    fn bitand(self, o: Key4) -> Key4 {
        Key4(std::array::from_fn(|i| self.0[i] & o.0[i]))
    }
}

impl PackedKey for Key4 {
    #[inline]
    fn mix(self, seed: u64) -> u64 {
        self.0.iter().fold(seed, |h, &w| crate::search::mix64(h ^ w))
    }
}

#[inline]
fn field(c: usize) -> (usize, u32) {
    (c / 12, 5 * (c % 12) as u32)
}

/// Which part of each cubie a search key looks at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Ignore,
    Position,
    Full,
}

fn key_mask4(parts: &[Part; N_CUBIES4]) -> Key4 {
    let mut k = Key4::default();
    for (c, &p) in parts.iter().enumerate() {
        let bits = match (p, piece4(c)) {
            (Part::Ignore, _) => 0,
            (Part::Position, Piece4::Corner) => 0b00111,
            _ => 0b11111,
        };
        let (w, sh) = field(c);
        k.0[w] |= bits << sh;
    }
    k
}

/// Per piece type [corners, wings, centers], the number of cubies whose packed fields differ.
fn diff_counts4(a: Key4, b: Key4) -> [u8; 3] {
    let mut n = [0u8; 3];
    for c in 0..N_CUBIES4 {
        let (w, sh) = field(c);
        if (a.0[w] ^ b.0[w]) >> sh & 0b11111 != 0 {
            n[piece4(c) as usize] += 1;
        }
    }
    n
}

// ---- bandaging ----

/// A set of blocks of a 4x4x4; all cubies of a block always move together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bandage4 {
    pub blocks: Vec<u64>,
}

impl Bandage4 {
    pub fn none() -> Bandage4 {
        Bandage4::default()
    }

    /// Build from groups of cubie names. Overlapping groups are merged.
    pub fn from_names<S: AsRef<str>>(groups: &[Vec<S>]) -> Result<Bandage4, String> {
        let mut masks = Vec::new();
        for g in groups {
            let mut m = 0u64;
            for name in g {
                let c = parse_cubie4(name.as_ref())
                    .ok_or_else(|| format!("unknown 4x4x4 cubie '{}'", name.as_ref()))?;
                m |= 1 << c;
            }
            masks.push(m);
        }
        Ok(Bandage4::from_masks(masks))
    }

    /// Parse a textual bandage, same syntax as the 3x3x3: `"UFR+UFr+URf | Ufr+Ufl"`.
    pub fn parse(spec: &str) -> Result<Bandage4, String> {
        let groups: Vec<Vec<&str>> = spec
            .split(['|', ';', ',', '\n'])
            .map(|blk| blk.split([' ', '+', '-', '\t']).filter(|t| !t.is_empty()).collect::<Vec<_>>())
            .filter(|g| !g.is_empty())
            .collect();
        Bandage4::from_names(&groups)
    }

    pub fn from_masks(mut masks: Vec<u64>) -> Bandage4 {
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..masks.len() {
                for j in i + 1..masks.len() {
                    if masks[i] & masks[j] != 0 {
                        masks[i] |= masks[j];
                        masks.swap_remove(j);
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }
        masks.retain(|m| m.count_ones() >= 2);
        masks.sort();
        Bandage4 { blocks: masks }
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Mask of all bandaged cubies.
    pub fn bandaged_cubies(&self) -> u64 {
        self.blocks.iter().fold(0, |a, b| a | b)
    }

    /// A turn of base `b` is legal if no block is cut by the turning layers.
    #[inline]
    pub fn allows(&self, s: &State4, b: usize) -> bool {
        if self.blocks.is_empty() {
            return true;
        }
        let lm = s.layer_mask(b);
        self.blocks.iter().all(|&blk| {
            let m = lm & blk;
            m == 0 || m == blk
        })
    }

    /// Every bandaged cubie is home and unrotated: the puzzle has its original shape.
    pub fn restored(&self, s: &State4) -> bool {
        let b = self.bandaged_cubies();
        (0..N_CUBIES4).all(|c| b & (1 << c) == 0 || s.rot[c] == ID_ROT)
    }

    /// The same position with the free centers of each colour relabelled: a free center on the
    /// home of a center of its colour is that center, the others fill the remaining slots in
    /// order. Two states look the same exactly when their canonical forms are equal.
    pub fn canonical(&self, s: &State4) -> State4 {
        let geo = geo4();
        let glued = self.bandaged_cubies();
        let mut out = *s;
        for f in 0..6 {
            let first = CENTER0_4 + 4 * f;
            let mut slots = [0u8; 4];
            let mut cubies = [0usize; 4];
            let mut n = 0;
            for c in first..first + 4 {
                if glued & (1 << c) == 0 {
                    slots[n] = s.slot(c) as u8;
                    cubies[n] = c;
                    n += 1;
                }
            }
            let mut taken = [false; 4];
            for &c in &cubies[..n] {
                if let Some(k) = slots[..n].iter().position(|&x| x as usize == c) {
                    taken[k] = true;
                }
            }
            let mut k = 0;
            for &c in &cubies[..n] {
                if slots[..n].iter().any(|&x| x as usize == c) {
                    out.rot[c] = ID_ROT;
                    continue;
                }
                while taken[k] {
                    k += 1;
                }
                taken[k] = true;
                out.rot[c] = geo.unpack[c][slots[k] as usize][0];
            }
        }
        out
    }

    /// Slots occupied by the cubies of block `blk` in state `s`.
    fn block_slots(&self, s: &State4, blk: u64) -> u64 {
        (0..N_CUBIES4).filter(|&c| blk & (1 << c) != 0).fold(0, |m, c| m | 1 << s.slot(c))
    }

    /// Block permutation of `s`: `perm[i] = j` when block i sits on the home slots of block j.
    /// None if the puzzle has a different shape.
    pub fn block_permutation(&self, s: &State4) -> Option<Vec<usize>> {
        self.blocks
            .iter()
            .map(|&b| {
                let at = self.block_slots(s, b);
                self.blocks.iter().position(|&h| h == at)
            })
            .collect()
    }

    pub fn shape_at_home(&self, s: &State4) -> bool {
        self.block_permutation(s).is_some()
    }

    /// Cycles of moved blocks, e.g. `(UFL+UFl ULB+ULb)`; empty if no block moved, `"?"` if the
    /// shape is not the home shape.
    pub fn describe_blocks(&self, s: &State4) -> String {
        let Some(perm) = self.block_permutation(s) else { return "?".into() };
        let names = self.block_names();
        let mut seen = vec![false; perm.len()];
        let mut out = Vec::new();
        for start in 0..perm.len() {
            if seen[start] || perm[start] == start {
                continue;
            }
            let mut cyc = Vec::new();
            let mut i = start;
            while !seen[i] {
                seen[i] = true;
                cyc.push(names[i].join("+"));
                i = perm[i];
            }
            out.push(format!("({})", cyc.join(" ")));
        }
        out.join(" ")
    }

    /// Textual form accepted by `parse` (empty for no bandage).
    pub fn spec(&self) -> String {
        self.block_names().iter().map(|b| b.join("+")).collect::<Vec<_>>().join(" | ")
    }

    pub fn block_names(&self) -> Vec<Vec<String>> {
        let names = &geo4().names;
        self.blocks
            .iter()
            .map(|&m| (0..N_CUBIES4).filter(|c| m & (1 << c) != 0).map(|c| names[c].to_string()).collect())
            .collect()
    }

    /// Pairs of adjacent facelets (same face) whose cubies belong to the same block.
    pub fn bonded_facelet_pairs(&self, s: &State4) -> Vec<(usize, usize)> {
        let occ = facelet_cubies4(s);
        let same = |a: usize, b: usize| self.blocks.iter().any(|&m| m & (1 << a) != 0 && m & (1 << b) != 0);
        let mut out = Vec::new();
        for f in 0..6 {
            for r in 0..4 {
                for c in 0..4 {
                    let i = f * 16 + r * 4 + c;
                    if c < 3 && same(occ[i], occ[i + 1]) {
                        out.push((i, i + 1));
                    }
                    if r < 3 && same(occ[i], occ[i + 4]) {
                        out.push((i, i + 4));
                    }
                }
            }
        }
        out
    }

    /// What `s` does relative to solved, seen through `canonical`: wing cycles are reported as
    /// edge cycles, centers only move when a colour changes place.
    pub fn effect(&self, s: &State4) -> Effect {
        let t = self.canonical(s);
        let mut e = Effect::default();
        let mut seen = [false; N_CUBIES4];
        for start in 0..N_CUBIES4 {
            if seen[start] {
                continue;
            }
            let mut cyc = Vec::new();
            let mut c = start;
            while !seen[c] {
                seen[c] = true;
                cyc.push((c, t.ori(c)));
                c = t.slot(c);
            }
            match (piece4(start), cyc.len()) {
                (Piece4::Corner, 1) if cyc[0].1 != 0 => e.twisted.push(cyc[0]),
                (_, 1) => {}
                (Piece4::Corner, _) => e.corner_cycles.push(cyc),
                (Piece4::Wing, _) => e.edge_cycles.push(cyc),
                (Piece4::Center, _) => e.center_cycles.push(cyc.iter().map(|x| x.0).collect()),
            }
        }
        e
    }
}

/// A (possibly bandaged) 4x4x4.
#[derive(Clone, Debug, Default)]
pub struct Cube4 {
    pub state: State4,
    pub bandage: Bandage4,
}

impl Cube4 {
    pub fn new(bandage: Bandage4) -> Cube4 {
        Cube4 { state: State4::solved(), bandage }
    }

    pub fn can_move(&self, m: Move) -> bool {
        self.bandage.allows(&self.state, base_of(m))
    }

    /// Apply a sequence, checking the bandage before each move. On an illegal move the cube is
    /// left unchanged and the error names the move and its index.
    pub fn apply(&mut self, alg: &[Move]) -> Result<(), String> {
        let mut s = self.state;
        for (i, &m) in alg.iter().enumerate() {
            if !self.bandage.allows(&s, base_of(m)) {
                return Err(format!("move {} ({}) is blocked by the bandage", i + 1, move_name4(m)));
            }
            s.apply_move(m);
        }
        self.state = s;
        Ok(())
    }

    pub fn bandage_restored(&self) -> bool {
        self.bandage.restored(&self.state)
    }

    /// Every sticker shows its face's colour and the puzzle has its shape.
    pub fn is_solved(&self) -> bool {
        self.bandage.canonical(&self.state) == State4::solved()
    }

    pub fn legal_moves(&self, moves: &[Move]) -> Vec<Move> {
        moves.iter().copied().filter(|&m| self.can_move(m)).collect()
    }

    pub fn effect(&self) -> Effect {
        self.bandage.effect(&self.state)
    }
}

fn key_part4(kind: Kind, p: Piece4) -> Part {
    match (kind, p) {
        (Kind::Any, _) => Part::Ignore,
        (Kind::Edge3, Piece4::Wing) | (Kind::Corner3, Piece4::Corner) | (Kind::Center3, Piece4::Center) => Part::Ignore,
        (Kind::CornerTwist | Kind::Orient, Piece4::Corner) => Part::Position,
        _ => Part::Full,
    }
}

impl Puzzle for Bandage4 {
    type State = State4;
    type Key = Key4;

    fn solved(&self) -> State4 {
        State4::solved()
    }

    #[inline]
    fn apply_move(s: &mut State4, m: Move) {
        s.apply_move(m);
    }

    #[inline]
    fn allows(&self, s: &State4, b: usize) -> bool {
        Bandage4::allows(self, s, b)
    }

    fn pack(&self, s: &State4) -> Key4 {
        let t = self.canonical(s);
        let mut k = Key4::default();
        for c in 0..N_CUBIES4 {
            let slot = t.slot(c) as u64;
            let code = match piece4(c) {
                Piece4::Corner => slot | (t.ori(c) as u64) << 3,
                Piece4::Wing => slot - WING0 as u64,
                Piece4::Center => slot - CENTER0_4 as u64,
            };
            let (w, sh) = field(c);
            k.0[w] |= code << sh;
        }
        k
    }

    fn unpack(&self, k: Key4) -> State4 {
        let geo = geo4();
        let mut s = State4::default();
        for c in 0..N_CUBIES4 {
            let (w, sh) = field(c);
            let v = (k.0[w] >> sh & 0b11111) as usize;
            let (slot, ori) = match piece4(c) {
                Piece4::Corner => (v & 7, v >> 3),
                Piece4::Wing => (v + WING0, 0),
                Piece4::Center => (v + CENTER0_4, 0),
            };
            s.rot[c] = geo.unpack[c][slot][ori];
        }
        s
    }

    fn home_shape(&self, s: &State4) -> bool {
        self.shape_at_home(s)
    }

    #[inline]
    fn axis_of_base(b: usize) -> usize {
        b / PER_AXIS
    }

    fn simplify(alg: &[Move]) -> Alg {
        simplify4(alg)
    }

    fn format_alg(alg: &[Move]) -> String {
        format_alg4(alg)
    }

    fn lengths(alg: &[Move]) -> (usize, usize) {
        (length4(alg), qtm_length4(alg))
    }

    fn key_mask(&self, q: &Query) -> Result<Key4, String> {
        if q.kind == Kind::EdgeFlip {
            return Err("kind 'edge_flip' does not exist on a 4x4x4: wings cannot flip in place".into());
        }
        let pinned = if q.move_blocks { 0 } else { self.bandaged_cubies() };
        let free_mask = q.pieces.unwrap_or(u64::MAX) & !pinned;
        let parts: [Part; N_CUBIES4] = std::array::from_fn(|c| {
            if free_mask & (1 << c) != 0 { key_part4(q.kind, piece4(c)) } else { Part::Full }
        });
        Ok(key_mask4(&parts))
    }

    fn diff_ok(kind: Kind, a: Key4, b: Key4) -> bool {
        let d = diff_counts4(a, b);
        match kind {
            Kind::Edge3 => d == [0, 3, 0],
            Kind::Corner3 => d == [3, 0, 0],
            Kind::CornerTwist => d == [2, 0, 0],
            // free centers are relabelled by colour, so their count is only a hint
            Kind::Center3 => d[0] == 0 && d[1] == 0 && d[2] > 0,
            Kind::Orient => d[1] == 0 && d[2] == 0 && d[0] > 0 && d[0] <= 2,
            Kind::EdgeFlip => false,
            Kind::Any => d != [0, 0, 0],
        }
    }

    fn verify(&self, alg: &[Move], q: &Query) -> Result<Effect, Reject> {
        let mut cube = Cube4::new(self.clone());
        cube.apply(alg).map_err(|_| Reject::Illegal)?;
        let shape_ok = if q.move_blocks { self.shape_at_home(&cube.state) } else { cube.bandage_restored() };
        if !shape_ok {
            return Err(Reject::NoMatch);
        }
        let effect = cube.effect();
        if !q.kind.matches(&effect, q.pure) {
            return Err(Reject::NoMatch);
        }
        if let Some(p) = q.pieces
            && effect.affected().iter().any(|&c| p & (1 << c) == 0)
        {
            return Err(Reject::NoMatch);
        }
        Ok(effect)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Solver;

    fn after(alg: &str) -> State4 {
        let mut s = State4::solved();
        s.apply_alg(&parse_alg4(alg).unwrap());
        s
    }

    fn solved(alg: &str) -> bool {
        Bandage4::none().canonical(&after(alg)) == State4::solved()
    }

    #[test]
    fn names() {
        let geo = geo4();
        for (c, n) in geo.names.iter().enumerate() {
            assert_eq!(parse_cubie4(n), Some(c), "{}", n);
        }
        assert_eq!(parse_cubie4("UFR"), Some(0));
        assert_eq!(geo.names[WING0..WING0 + 4], ["URf", "URb", "UFr", "UFl"]);
        assert_eq!(geo.names[CENTER0_4..CENTER0_4 + 4], ["Ubl", "Ubr", "Ufl", "Ufr"]);
        assert_eq!(parse_cubie4("rFU"), parse_cubie4("UFr"));
        assert_eq!(parse_cubie4("ufr"), None); // inner core
        assert_eq!(parse_cubie4("UF"), None);
    }

    #[test]
    fn notation() {
        let a = parse_alg4("R 2R' Rw r2 l' x y' z2 2F (U 2D)2").unwrap();
        assert_eq!(format_alg4(&a), "R 2R' r r2 l' x y' z2 2F U 2D U 2D");
        assert!(parse_alg4("M").is_err());
        assert!(parse_alg4("2r").is_err());
        assert_eq!(format_alg4(&simplify4(&parse_alg4("R 2R R' 2L 2R'").unwrap())), "2L");
        assert_eq!(parse_moveset4("htm").unwrap().len(), 18);
        assert_eq!(parse_moveset4("stm").unwrap().len(), 36);
        assert_eq!(parse_moveset4("R 2R2").unwrap().len(), 4);
        assert_eq!((length4(&a), qtm_length4(&a)), (10, 11));
    }

    #[test]
    fn facelets_and_identities() {
        assert_eq!(State4::solved().facelets(), ["U", "R", "F", "D", "L", "B"].map(|f| f.repeat(16)).concat());
        let r = after("R").facelets();
        assert_eq!(&r[0..16], "UUUF".repeat(4));
        assert_eq!(&r[32..48], "FFFD".repeat(4));
        assert_eq!(&r[80..96], "UBBB".repeat(4));
        let r2 = after("2R").facelets();
        assert_eq!(&r2[0..16], "UUFU".repeat(4));
        assert!(solved("(R U R' U')6"));
        assert!(solved("(R U)105"));
        assert!(solved("r R' 2R'"));
        assert!(solved("x L 2L R' 2R'"));
        assert!(solved("u U' 2U'"));
        assert!(solved("(2R)4"));
        assert!(!solved("2R2 U2 2R2 U2"));
        // U spins the four U centers: invisible, unless they are glued
        assert!(Bandage4::none().effect(&after("U")).center_cycles.is_empty());
        assert_eq!(Bandage4::parse("Ufr+Ufl").unwrap().effect(&after("U")).center_cycles.len(), 1);
    }

    #[test]
    fn parity_is_a_wing_swap() {
        // OLL parity: two wings of one composite edge exchanged, centers look solved
        let s = after("2R2 B2 U2 2L U2 2R' U2 2R U2 F2 2R F2 2L' B2 2R2");
        let e = Bandage4::none().effect(&s);
        assert_eq!(e.n_corners(), 0, "{}", e.describe_with(&geo4().names));
        assert!(e.center_cycles.is_empty(), "{}", e.describe_with(&geo4().names));
        assert_eq!(e.edge_cycles.len(), 1);
        assert_eq!(e.edge_cycles[0].len(), 2);
    }

    #[test]
    fn pack_roundtrip() {
        let b = Bandage4::parse("UFR+UFr | Dfr+Dfl").unwrap();
        let s = after("R 2R' U F2 2D l b' 2B");
        let k = b.pack(&s);
        let u = b.unpack(k);
        assert_eq!(b.pack(&u), k);
        assert_eq!(u.facelets(), s.facelets());
        assert_eq!(b.canonical(&u), b.canonical(&s));
    }

    #[test]
    fn bandage_blocks_moves() {
        let b = Bandage4::parse("UFR+UFr+URf").unwrap();
        let mut c = Cube4::new(b.clone());
        assert!(!c.can_move(parse_alg4("R").unwrap()[0]));
        assert!(!c.can_move(parse_alg4("2R").unwrap()[0]));
        assert!(c.can_move(parse_alg4("r").unwrap()[0]));
        assert!(!c.can_move(parse_alg4("F").unwrap()[0])); // UFR and UFr in F, URf not
        assert!(c.apply(&parse_alg4("U r' D").unwrap()).is_ok());
        assert!(c.apply(&parse_alg4("2U 2U'").unwrap()).is_ok());
        assert!(c.apply(&parse_alg4("D' r U'").unwrap()).is_ok());
        assert!(c.bandage_restored() && c.is_solved());
        assert_eq!(b.bonded_facelet_pairs(&State4::solved()).len(), 4); // U: 2, F: 1, R: 1
        assert!(Bandage4::parse("UFR+UFx").is_err());
        assert_eq!(Bandage4::parse(&b.spec()).unwrap(), b);
    }

    #[test]
    fn finds_wing_and_center_3cycles() {
        let s = Solver::new(Bandage4::none(), parse_moveset4("stm").unwrap(), 4, 10_000_000);
        assert_eq!(&s.layer_sizes()[..2], &[1, 36]);
        let names = &geo4().names;
        for kind in [Kind::Edge3, Kind::Center3, Kind::Corner3] {
            let (found, _) = s.search(&Query { max_results: 5, ..Query::new(kind) }).unwrap();
            assert!(!found.is_empty(), "{:?}", kind);
            assert_eq!(found[0].length, 8, "{:?}", kind);
            for f in &found {
                let e = Bandage4::none().effect(&{
                    let mut t = State4::solved();
                    t.apply_alg(&f.alg);
                    t
                });
                assert!(kind.matches(&e, false), "{} {}", format_alg4(&f.alg), e.describe_with(names));
            }
        }
        assert!(s.search(&Query::new(Kind::EdgeFlip)).is_err());
    }

    #[test]
    fn bandaged_results_are_legal() {
        let b = Bandage4::parse("UFR+UFr+UFl+UFL | Dfr+Dfl+Dbr+Dbl").unwrap();
        let s = Solver::new(b.clone(), parse_moveset4("stm").unwrap(), 4, 10_000_000);
        let (found, _) = s.search(&Query { max_results: 20, ..Query::new(Kind::Edge3) }).unwrap();
        assert!(!found.is_empty());
        for f in &found {
            let mut c = Cube4::new(b.clone());
            c.apply(&f.alg).unwrap();
            assert!(c.bandage_restored());
            assert!(Kind::Edge3.matches(&c.effect(), false));
        }
    }
}
