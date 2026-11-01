//! Cube state, bandaging and effect analysis.

use crate::geometry::{
    face_of_normal, geom, parse_cubie, piece_type, PieceType, Vec3, CENTER0, EDGE0, FACES, FACE_NORMALS, ID_ROT,
    NAMES, N_CUBIES,
};
use crate::moves::{base_of, move_name, tables, Move};
use std::fmt::Write as _;
use std::sync::OnceLock;

/// Rotation of every cubie relative to its home.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub rot: [u8; N_CUBIES],
}

impl Default for State {
    fn default() -> Self {
        State { rot: [ID_ROT; N_CUBIES] }
    }
}

// ---- packing into a u128 (corners 5 bits, edges 5 bits, centers 3 bits = 118 bits) ----

struct PackTables {
    offset: [u32; N_CUBIES],
    /// code[c][g]: bit pattern of cubie c with rotation g (slot bits | ori bits)
    code: [[u8; 24]; N_CUBIES],
    field_mask: [u128; N_CUBIES],
    pos_mask: [u128; N_CUBIES],
}

static PACK: OnceLock<PackTables> = OnceLock::new();

fn pack_tables() -> &'static PackTables {
    PACK.get_or_init(|| {
        let geo = geom();
        let mut t = PackTables {
            offset: [0; N_CUBIES],
            code: [[0; 24]; N_CUBIES],
            field_mask: [0; N_CUBIES],
            pos_mask: [0; N_CUBIES],
        };
        let mut off = 0u32;
        for c in 0..N_CUBIES {
            let (base, slot_bits, width) = match piece_type(c) {
                PieceType::Corner => (0, 3, 5),
                PieceType::Edge => (EDGE0, 4, 5),
                PieceType::Center => (CENTER0, 3, 3),
            };
            t.offset[c] = off;
            t.field_mask[c] = ((1u128 << width) - 1) << off;
            t.pos_mask[c] = ((1u128 << slot_bits) - 1) << off;
            for g in 0..24 {
                let slot = geo.slot[c][g] as usize - base;
                t.code[c][g] = (slot as u8) | (geo.ori[c][g] << slot_bits);
            }
            off += width;
        }
        t
    })
}

/// Which part of each cubie a search key looks at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyPart {
    Ignore,
    Position,
    Full,
}

pub fn key_mask(parts: &[KeyPart; N_CUBIES]) -> u128 {
    let t = pack_tables();
    (0..N_CUBIES)
        .map(|c| match parts[c] {
            KeyPart::Ignore => 0,
            KeyPart::Position => t.pos_mask[c],
            KeyPart::Full => t.field_mask[c],
        })
        .fold(0, |a, b| a | b)
}

/// Per piece type, the number of cubies whose packed fields differ.
pub fn diff_counts(a: u128, b: u128) -> [u8; 3] {
    let t = pack_tables();
    let x = a ^ b;
    let mut n = [0u8; 3];
    if x == 0 {
        return n;
    }
    for c in 0..N_CUBIES {
        if x & t.field_mask[c] != 0 {
            n[piece_type(c) as usize] += 1;
        }
    }
    n
}

impl State {
    pub fn solved() -> State {
        State::default()
    }

    #[inline]
    pub fn apply_move(&mut self, m: Move) {
        let next = &tables().next[m as usize];
        for c in 0..N_CUBIES {
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
    pub fn layer_mask(&self, b: usize) -> u32 {
        let lb = &tables().layer_bit[b];
        let mut mask = 0;
        for c in 0..N_CUBIES {
            mask |= lb[c][self.rot[c] as usize];
        }
        mask
    }

    pub fn pack(&self) -> u128 {
        let t = pack_tables();
        let mut x = 0u128;
        for c in 0..N_CUBIES {
            x |= (t.code[c][self.rot[c] as usize] as u128) << t.offset[c];
        }
        x
    }

    pub fn unpack(x: u128) -> State {
        let t = pack_tables();
        let geo = geom();
        let mut s = State::default();
        for c in 0..N_CUBIES {
            let v = ((x & t.field_mask[c]) >> t.offset[c]) as u8;
            let (base, slot_bits) = match piece_type(c) {
                PieceType::Corner => (0, 3),
                PieceType::Edge => (EDGE0, 4),
                PieceType::Center => (CENTER0, 3),
            };
            let slot = (v & ((1 << slot_bits) - 1)) as usize + base;
            let ori = (v >> slot_bits) as usize;
            s.rot[c] = geo.unpack[c][slot][ori];
        }
        s
    }

    /// Current slot (index of the home cubie of that position) of cubie `c`.
    pub fn slot(&self, c: usize) -> usize {
        geom().slot[c][self.rot[c] as usize] as usize
    }

    pub fn ori(&self, c: usize) -> u8 {
        geom().ori[c][self.rot[c] as usize]
    }

    /// Cubie occupying each slot.
    pub fn occupants(&self) -> [usize; N_CUBIES] {
        let mut occ = [0; N_CUBIES];
        for c in 0..N_CUBIES {
            occ[self.slot(c)] = c;
        }
        occ
    }

    /// Solved = every corner/edge at home and oriented, every center at home (spin ignored).
    pub fn is_solved(&self) -> bool {
        (0..N_CUBIES).all(|c| self.slot(c) == c && self.ori(c) == 0)
    }

    /// 54-character facelet string in URFDLB order (Kociemba layout).
    pub fn facelets(&self) -> String {
        let occ = self.occupants();
        let geo = geom();
        let mut s = String::with_capacity(54);
        for i in 0..54 {
            let (p, n) = facelet_geometry(i);
            let slot = crate::geometry::cubie_at_position(p).unwrap();
            let c = occ[slot];
            let g = self.rot[c];
            let home_normal = geo.apply(geo.inv[g as usize], n);
            s.push(FACES[face_of_normal(home_normal)]);
        }
        s
    }

    pub fn effect(&self) -> Effect {
        Effect::of(self)
    }
}

/// Position of the cubie and outward normal of facelet `i` (0..54, URFDLB, row-major).
pub fn facelet_geometry(i: usize) -> (Vec3, Vec3) {
    let (f, r, c) = (i / 9, (i % 9 / 3) as i8, (i % 3) as i8);
    let p = match f {
        0 => [c - 1, 1, r - 1],
        1 => [1, 1 - r, 1 - c],
        2 => [c - 1, 1 - r, 1],
        3 => [c - 1, -1, 1 - r],
        4 => [-1, 1 - r, c - 1],
        _ => [1 - c, 1 - r, -1],
    };
    (p, FACE_NORMALS[f])
}

/// Cubie currently showing each of the 54 facelets.
pub fn facelet_cubies(s: &State) -> [usize; 54] {
    let occ = s.occupants();
    let mut out = [0; 54];
    for (i, o) in out.iter_mut().enumerate() {
        *o = occ[crate::geometry::cubie_at_position(facelet_geometry(i).0).unwrap()];
    }
    out
}

// ---- bandaging ----

/// A set of blocks; all cubies of a block always move together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bandage {
    pub blocks: Vec<u32>,
}

impl Bandage {
    pub fn none() -> Bandage {
        Bandage::default()
    }

    /// Build from groups of cubie names. Overlapping groups are merged.
    pub fn from_names<S: AsRef<str>>(groups: &[Vec<S>]) -> Result<Bandage, String> {
        let mut masks = Vec::new();
        for g in groups {
            let mut m = 0u32;
            for name in g {
                let c = parse_cubie(name.as_ref()).ok_or_else(|| format!("unknown cubie '{}'", name.as_ref()))?;
                m |= 1 << c;
            }
            masks.push(m);
        }
        Ok(Bandage::from_masks(masks))
    }

    /// Parse a textual bandage: blocks separated by `|`, `;` or `,`, cubies inside a block by
    /// spaces, `+` or `-`. Example: `"UF+UFR+UR | D DF"`.
    pub fn parse(spec: &str) -> Result<Bandage, String> {
        let groups: Vec<Vec<&str>> = spec
            .split(['|', ';', ',', '\n'])
            .map(|blk| blk.split([' ', '+', '-', '\t']).filter(|t| !t.is_empty()).collect::<Vec<_>>())
            .filter(|g| !g.is_empty())
            .collect();
        Bandage::from_names(&groups)
    }

    pub fn from_masks(mut masks: Vec<u32>) -> Bandage {
        // merge overlapping blocks until fixpoint
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
        Bandage { blocks: masks }
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Mask of all bandaged cubies.
    pub fn bandaged_cubies(&self) -> u32 {
        self.blocks.iter().fold(0, |a, b| a | b)
    }

    /// A turn of base `b` is legal if no block is cut by the turning layers.
    #[inline]
    pub fn allows(&self, s: &State, b: usize) -> bool {
        if self.blocks.is_empty() {
            return true;
        }
        let lm = s.layer_mask(b);
        self.blocks.iter().all(|&blk| {
            let m = lm & blk;
            m == 0 || m == blk
        })
    }

    /// Slots occupied by the cubies of block `blk` in state `s`.
    fn block_slots(&self, s: &State, blk: u32) -> u32 {
        (0..N_CUBIES).filter(|&c| blk & (1 << c) != 0).fold(0, |m, c| m | 1 << s.slot(c))
    }

    /// Block permutation of `s`: `perm[i] = j` when block i sits on the home slots of block j
    /// (a block that fills another block's slots is congruent to it). None if some block is
    /// not on any block's home, i.e. the puzzle has a different shape.
    pub fn block_permutation(&self, s: &State) -> Option<Vec<usize>> {
        self.blocks
            .iter()
            .map(|&b| {
                let at = self.block_slots(s, b);
                self.blocks.iter().position(|&h| h == at)
            })
            .collect()
    }

    /// The puzzle has its home shape, possibly with blocks exchanged for congruent ones.
    pub fn shape_at_home(&self, s: &State) -> bool {
        self.block_permutation(s).is_some()
    }

    /// Cycles of moved blocks, e.g. `(UFL+UF ULB+UB)`: the block at the first home moves to the
    /// second. Empty if no block moved, `"?"` if the shape is not the home shape.
    pub fn describe_blocks(&self, s: &State) -> String {
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

    /// Textual form accepted by `parse`, e.g. `"UFR+UF | DF+D"` (empty for no bandage).
    pub fn spec(&self) -> String {
        self.block_names().iter().map(|b| b.join("+")).collect::<Vec<_>>().join(" | ")
    }

    pub fn block_names(&self) -> Vec<Vec<String>> {
        self.blocks
            .iter()
            .map(|&m| (0..N_CUBIES).filter(|c| m & (1 << c) != 0).map(|c| NAMES[c].to_string()).collect())
            .collect()
    }

    /// Pairs of adjacent facelets (same face) whose cubies belong to the same block.
    pub fn bonded_facelet_pairs(&self, s: &State) -> Vec<(usize, usize)> {
        let occ = facelet_cubies(s);
        let cubie = |i: usize| occ[i];
        let same = |a: usize, b: usize| self.blocks.iter().any(|&m| m & (1 << a) != 0 && m & (1 << b) != 0);
        let mut out = Vec::new();
        for f in 0..6 {
            for r in 0..3 {
                for c in 0..3 {
                    let i = f * 9 + r * 3 + c;
                    if c < 2 && same(cubie(i), cubie(i + 1)) {
                        out.push((i, i + 1));
                    }
                    if r < 2 && same(cubie(i), cubie(i + 3)) {
                        out.push((i, i + 3));
                    }
                }
            }
        }
        out
    }
}

/// A (possibly bandaged) cube.
#[derive(Clone, Debug, Default)]
pub struct Cube {
    pub state: State,
    pub bandage: Bandage,
}

impl Cube {
    pub fn new(bandage: Bandage) -> Cube {
        Cube { state: State::solved(), bandage }
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
                return Err(format!("move {} ({}) is blocked by the bandage", i + 1, move_name(m)));
            }
            s.apply_move(m);
        }
        self.state = s;
        Ok(())
    }

    /// Every bandaged cubie is home and unrotated (a bandaged center's spin is visible through
    /// the block it belongs to), so the puzzle has its original shape.
    pub fn bandage_restored(&self) -> bool {
        let b = self.bandage.bandaged_cubies();
        (0..N_CUBIES).all(|c| b & (1 << c) == 0 || self.state.rot[c] == ID_ROT)
    }

    /// Solved, including the shape: bandaged centers must not be spun.
    pub fn is_solved(&self) -> bool {
        self.state.is_solved() && self.bandage_restored()
    }

    pub fn legal_moves(&self, moves: &[Move]) -> Vec<Move> {
        moves.iter().copied().filter(|&m| self.can_move(m)).collect()
    }
}

// ---- effect analysis ----

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Effect {
    /// Cycles of cubies: each entry is (cubie, orientation at its destination). The piece starting at
    /// the slot of entry k moves to the slot of entry k+1.
    pub edge_cycles: Vec<Vec<(usize, u8)>>,
    pub corner_cycles: Vec<Vec<(usize, u8)>>,
    pub center_cycles: Vec<Vec<usize>>,
    /// Edges flipped in place.
    pub flipped: Vec<usize>,
    /// Corners twisted in place: (cubie, 1 = clockwise, 2 = counter-clockwise).
    pub twisted: Vec<(usize, u8)>,
}

impl Effect {
    pub fn of(s: &State) -> Effect {
        let mut e = Effect::default();
        let mut seen = [false; N_CUBIES];
        for start in 0..N_CUBIES {
            if seen[start] {
                continue;
            }
            let mut cyc = Vec::new();
            let mut c = start;
            while !seen[c] {
                seen[c] = true;
                cyc.push((c, s.ori(c)));
                c = s.slot(c);
            }
            match (piece_type(start), cyc.len()) {
                (PieceType::Edge, 1) if cyc[0].1 != 0 => e.flipped.push(start),
                (PieceType::Corner, 1) if cyc[0].1 != 0 => e.twisted.push(cyc[0]),
                (_, 1) => {}
                (PieceType::Edge, _) => e.edge_cycles.push(cyc),
                (PieceType::Corner, _) => e.corner_cycles.push(cyc),
                (PieceType::Center, _) => e.center_cycles.push(cyc.iter().map(|x| x.0).collect()),
            }
        }
        e
    }

    pub fn is_identity(&self) -> bool {
        self.n_edges() == 0 && self.n_corners() == 0 && self.center_cycles.is_empty()
    }

    pub fn n_edges(&self) -> usize {
        self.flipped.len() + self.edge_cycles.iter().map(Vec::len).sum::<usize>()
    }

    pub fn n_corners(&self) -> usize {
        self.twisted.len() + self.corner_cycles.iter().map(Vec::len).sum::<usize>()
    }

    /// Every affected corner/edge (moved or reoriented) and every moved center.
    pub fn affected(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.edge_cycles.iter().flatten().map(|x| x.0).collect();
        v.extend(&self.flipped);
        v.extend(self.corner_cycles.iter().flatten().map(|x| x.0));
        v.extend(self.twisted.iter().map(|x| x.0));
        v.extend(self.center_cycles.iter().flatten());
        v.sort();
        v
    }

    /// True if every moved corner/edge keeps its orientation (a "pure" cycle).
    pub fn is_pure(&self) -> bool {
        self.edge_cycles.iter().chain(&self.corner_cycles).flatten().all(|x| x.1 == 0)
    }

    /// Human readable form, e.g. `(UF UR UB)`, `(URF UBR+ DFR-)`, `UF* UB*`, `URF+ UBR-`.
    /// `*` marks a flipped edge, `+`/`-` a corner twisted clockwise/counter-clockwise.
    pub fn describe(&self) -> String {
        self.describe_with(&NAMES)
    }

    /// `describe` with the cubie names of another cube size.
    pub fn describe_with(&self, names: &[&str]) -> String {
        let edge = |c: usize, o: u8| format!("{}{}", names[c], if o == 0 { "" } else { "*" });
        let corner = |c: usize, o: u8| format!("{}{}", names[c], ["", "+", "-"][o as usize]);
        let cycle = |cyc: &Vec<(usize, u8)>, mark: &dyn Fn(usize, u8) -> String| {
            format!("({})", cyc.iter().map(|&(c, o)| mark(c, o)).collect::<Vec<_>>().join(" "))
        };
        let mut parts: Vec<String> = Vec::new();
        parts.extend(self.edge_cycles.iter().map(|c| cycle(c, &edge)));
        parts.extend(self.corner_cycles.iter().map(|c| cycle(c, &corner)));
        parts.extend(self.flipped.iter().map(|&c| edge(c, 1)));
        parts.extend(self.twisted.iter().map(|&(c, o)| corner(c, o)));
        for cyc in &self.center_cycles {
            let mut s = String::from("(");
            for (i, &c) in cyc.iter().enumerate() {
                let _ = write!(s, "{}{}", if i > 0 { " " } else { "" }, names[c]);
            }
            s.push(')');
            parts.push(s);
        }
        if parts.is_empty() { "identity".to_string() } else { parts.join(" ") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::parse_alg;

    fn after(alg: &str) -> State {
        let mut s = State::solved();
        s.apply_alg(&parse_alg(alg).unwrap());
        s
    }

    #[test]
    fn facelets_solved_and_moves() {
        assert_eq!(
            State::solved().facelets(),
            "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
        );
        // Known facelet strings for single moves (Kociemba layout).
        assert_eq!(after("R").facelets(), "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB");
        assert_eq!(after("U").facelets(), "UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB");
        assert_eq!(after("F").facelets(), "UUUUUULLLURRURRURRFFFFFFFFFRRRDDDDDDLLDLLDLLDBBBBBBBBB");
    }

    #[test]
    fn order_and_identities() {
        assert!(after("R U R' U'").pack() != State::solved().pack());
        assert!(after("(R U R' U')6").is_solved());
        assert!(after("x y z").facelets() != State::solved().facelets());
        assert!(after("(R U)105").is_solved());
        assert!(after("r R' M").is_solved()); // r = R M'
        assert!(after("u U' E").is_solved());
        assert!(after("f F' S'").is_solved());
        assert!(after("x R' L M").is_solved());
    }

    #[test]
    fn pack_roundtrip() {
        let s = after("R U F' L2 D B' M E' S r");
        let u = State::unpack(s.pack());
        assert_eq!(u.pack(), s.pack());
        assert_eq!(u.facelets(), s.facelets());
    }

    #[test]
    fn effects() {
        // T-perm swaps UL/UR and UBR/URF; sune twists 3 corners & permutes; check 3-cycle
        let u_perm = after("M2 U M U2 M' U M2");
        let e = u_perm.effect();
        assert_eq!(e.edge_cycles.len(), 1);
        assert_eq!(e.n_edges(), 3);
        assert_eq!(e.n_corners(), 0);
        assert!(e.is_pure());
        let a_perm = after("R' F R' B2 R F' R' B2 R2");
        let e = a_perm.effect();
        assert_eq!(e.corner_cycles.len(), 1);
        assert_eq!(e.n_corners(), 3);
        assert_eq!(e.n_edges(), 0);
        let flip = after("M' U M' U M' U2 M U M U M U2");
        let e = flip.effect();
        assert_eq!(e.flipped.len(), 2, "{}", e.describe());
        let twist = after("(R' D' R D)2 U (D' R' D R)2 U'");
        let e = twist.effect();
        assert_eq!(e.twisted.len(), 2, "{}", e.describe());
        assert_eq!(e.n_edges(), 0);
    }

    #[test]
    fn bandage_blocks_moves() {
        let b = Bandage::from_names(&[vec!["UF", "UFR"]]).unwrap();
        let mut c = Cube::new(b);
        assert!(c.apply(&parse_alg("U").unwrap()).is_ok());
        assert!(c.apply(&parse_alg("U'").unwrap()).is_ok());
        assert!(c.apply(&parse_alg("R").unwrap()).is_err()); // UFR in R, UF not
        assert!(c.apply(&parse_alg("L D B M'").unwrap()).is_err()); // M cuts the block
        assert!(c.apply(&parse_alg("L D B").unwrap()).is_ok());
        // a block bonded only to a center can wander around that face: shape not restored
        let mut d = Cube::new(Bandage::parse("DF+D").unwrap());
        d.apply(&parse_alg("D").unwrap()).unwrap();
        assert!(!d.bandage_restored());
        let pairs = c.bandage.bonded_facelet_pairs(&State::solved());
        assert_eq!(pairs.len(), 2); // on U and on F
        assert_eq!(Bandage::parse("FU+UFR | D-DF, R").unwrap().blocks.len(), 2);
        assert!(Bandage::parse("UF+QQ").is_err());
    }
}
