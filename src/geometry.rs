//! Static geometry of the 3x3x3 cube: the 24 rotations, the 26 cubies and their stickers.
//!
//! Coordinates: +x = R, +y = U, +z = F. Every cubie sits at a point of {-1,0,1}^3 (the core
//! at the origin is not modelled). A cubie's state is a single rotation (0..24) that maps its
//! home position/stickers to its current position/stickers.

use std::sync::OnceLock;

pub type Vec3 = [i8; 3];

pub const N_CUBIES: usize = 26;
pub const N_CORNERS: usize = 8;
pub const N_EDGES: usize = 12;
pub const N_CENTERS: usize = 6;
pub const CORNER0: usize = 0;
pub const EDGE0: usize = 8;
pub const CENTER0: usize = 20;
pub const N_ROT: usize = 24;
pub const ID_ROT: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceType {
    Corner,
    Edge,
    Center,
}

pub fn piece_type(c: usize) -> PieceType {
    if c < EDGE0 {
        PieceType::Corner
    } else if c < CENTER0 {
        PieceType::Edge
    } else {
        PieceType::Center
    }
}

/// Canonical cubie names (Kociemba order for corners and edges).
pub const NAMES: [&str; N_CUBIES] = [
    "URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB", // corners
    "UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR", // edges
    "U", "R", "F", "D", "L", "B", // centers
];

/// Face letters in facelet order (URFDLB) and their outward normals.
pub const FACES: [char; 6] = ['U', 'R', 'F', 'D', 'L', 'B'];
pub const FACE_NORMALS: [Vec3; 6] = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1]];

pub fn face_of_normal(n: Vec3) -> usize {
    FACE_NORMALS.iter().position(|&f| f == n).expect("not a face normal")
}

fn face_from_letter(ch: char) -> Option<usize> {
    FACES.iter().position(|&f| f == ch.to_ascii_uppercase())
}

/// A proper rotation of the cube as an integer 3x3 matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mat(pub [[i8; 3]; 3]);

impl Mat {
    const IDENTITY: Mat = Mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);

    fn mul(&self, o: &Mat) -> Mat {
        let mut r = [[0i8; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = (0..3).map(|k| self.0[i][k] * o.0[k][j]).sum();
            }
        }
        Mat(r)
    }

    fn apply(&self, v: Vec3) -> Vec3 {
        let mut r = [0i8; 3];
        for i in 0..3 {
            r[i] = (0..3).map(|k| self.0[i][k] * v[k]).sum();
        }
        r
    }

    /// Rotation by `q` quarter turns (counter-clockwise, right-hand rule) about `axis`.
    pub fn quarter(axis: usize, q: i32) -> Mat {
        let base = match axis {
            0 => Mat([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            1 => Mat([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            _ => Mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        };
        let mut m = Mat::IDENTITY;
        for _ in 0..q.rem_euclid(4) {
            m = base.mul(&m);
        }
        m
    }
}

/// All precomputed tables. Built once on first use.
pub struct Geometry {
    pub rots: Vec<Mat>,
    /// mul[a][b] = a∘b (apply b first, then a)
    pub mul: [[u8; N_ROT]; N_ROT],
    pub inv: [u8; N_ROT],
    pub home: [Vec3; N_CUBIES],
    /// slot[c][g]: index of the cubie whose home is where cubie `c` sits under rotation g.
    pub slot: [[u8; N_ROT]; N_CUBIES],
    /// ori[c][g]: orientation of cubie `c` under rotation g, using the usual U/D-F/B convention
    /// (corners 0..3, 1 = twisted clockwise; edges 0..2; centers always 0).
    pub ori: [[u8; N_ROT]; N_CUBIES],
    /// unpack[c][slot][ori] -> rotation (centers: a canonical rotation, spin is ignored).
    pub unpack: [[[u8; 3]; N_CUBIES]; N_CUBIES],
}

static GEOM: OnceLock<Geometry> = OnceLock::new();

pub fn geom() -> &'static Geometry {
    GEOM.get_or_init(Geometry::build)
}

pub fn cubie_at_position(p: Vec3) -> Option<usize> {
    geom().home.iter().position(|&h| h == p)
}

/// Parse a cubie name in any letter order ("UFR", "rfu", "FU", "U"...).
pub fn parse_cubie(name: &str) -> Option<usize> {
    let mut p = [0i8; 3];
    let mut seen = [false; 3];
    let name = name.trim();
    if name.is_empty() || name.len() > 3 {
        return None;
    }
    for ch in name.chars() {
        let f = face_from_letter(ch)?;
        let n = FACE_NORMALS[f];
        let axis = n.iter().position(|&v| v != 0).unwrap();
        if seen[axis] {
            return None;
        }
        seen[axis] = true;
        p[axis] = n[axis];
    }
    cubie_at_position(p)
}

/// Sticker normals of a cubie at home, reference sticker first.
/// Reference: the U/D sticker if any, else F/B; corners list the rest clockwise.
pub fn sticker_normals(c: usize) -> Vec<Vec3> {
    sticker_normals_with(&geom().home, c)
}

impl Geometry {
    fn build() -> Geometry {
        // Rotation group by closure over the three quarter turns.
        let gens = [Mat::quarter(0, 1), Mat::quarter(1, 1), Mat::quarter(2, 1)];
        let mut rots = vec![Mat::IDENTITY];
        let mut i = 0;
        while i < rots.len() {
            for g in &gens {
                let m = g.mul(&rots[i]);
                if !rots.contains(&m) {
                    rots.push(m);
                }
            }
            i += 1;
        }
        assert_eq!(rots.len(), N_ROT);
        let idx = |m: &Mat| rots.iter().position(|r| r == m).unwrap() as u8;
        let mut mul = [[0u8; N_ROT]; N_ROT];
        let mut inv = [0u8; N_ROT];
        for a in 0..N_ROT {
            for b in 0..N_ROT {
                mul[a][b] = idx(&rots[a].mul(&rots[b]));
                if mul[a][b] == ID_ROT {
                    inv[a] = b as u8;
                }
            }
        }

        let mut home = [[0i8; 3]; N_CUBIES];
        for (c, name) in NAMES.iter().enumerate() {
            for ch in name.chars() {
                let n = FACE_NORMALS[face_from_letter(ch).unwrap()];
                for k in 0..3 {
                    home[c][k] += n[k];
                }
            }
        }

        let mut geo = Geometry {
            rots,
            mul,
            inv,
            home,
            slot: [[0; N_ROT]; N_CUBIES],
            ori: [[0; N_ROT]; N_CUBIES],
            unpack: [[[0; 3]; N_CUBIES]; N_CUBIES],
        };
        let mut unpack_set = [[[false; 3]; N_CUBIES]; N_CUBIES];
        for c in 0..N_CUBIES {
            let own = sticker_normals_with(&geo.home, c);
            for g in 0..N_ROT {
                let p = geo.rots[g].apply(geo.home[c]);
                let s = geo.home.iter().position(|&h| h == p).unwrap();
                geo.slot[c][g] = s as u8;
                let o = if piece_type(c) == PieceType::Center {
                    0
                } else {
                    let slot_ns = sticker_normals_with(&geo.home, s);
                    let moved_ref = geo.rots[g].apply(own[0]);
                    slot_ns.iter().position(|&n| n == moved_ref).unwrap() as u8
                };
                geo.ori[c][g] = o;
                if !unpack_set[c][s][o as usize] {
                    unpack_set[c][s][o as usize] = true;
                    geo.unpack[c][s][o as usize] = g as u8;
                }
            }
        }
        geo
    }

    pub fn apply(&self, g: u8, v: Vec3) -> Vec3 {
        self.rots[g as usize].apply(v)
    }

    pub fn position(&self, c: usize, g: u8) -> Vec3 {
        self.apply(g, self.home[c])
    }
}

/// Reference sticker first (U/D, else F/B); corner stickers listed clockwise seen from outside.
fn sticker_normals_with(home: &[Vec3; N_CUBIES], c: usize) -> Vec<Vec3> {
    let h = home[c];
    let mut ns: Vec<Vec3> = Vec::new();
    for axis in [1usize, 2, 0] {
        if h[axis] != 0 {
            let mut n = [0i8; 3];
            n[axis] = h[axis];
            ns.push(n);
        }
    }
    if ns.len() == 3 {
        let (a, b) = (ns[0], ns[1]);
        let cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let dot: i8 = (0..3).map(|i| cross[i] * h[i]).sum();
        if dot > 0 {
            ns.swap(1, 2);
        }
    }
    ns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_tables() {
        let g = geom();
        for a in 0..N_ROT {
            assert_eq!(g.mul[a][g.inv[a] as usize], ID_ROT);
        }
        assert_eq!(g.rots[0], Mat::IDENTITY);
    }

    #[test]
    fn names_parse() {
        assert_eq!(parse_cubie("UFR"), Some(0));
        assert_eq!(parse_cubie("fu"), Some(9));
        assert_eq!(parse_cubie("B"), Some(25));
        assert_eq!(parse_cubie("UD"), None);
        for (c, n) in NAMES.iter().enumerate() {
            assert_eq!(parse_cubie(n), Some(c));
        }
    }

    #[test]
    fn corner_sticker_order_is_clockwise() {
        // URF: U, R, F (Kociemba facelet order)
        assert_eq!(sticker_normals(0), vec![[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
    }

    #[test]
    fn unpack_roundtrip() {
        let g = geom();
        for c in 0..N_CUBIES {
            for r in 0..N_ROT as u8 {
                let s = g.slot[c][r as usize];
                let o = g.ori[c][r as usize];
                let back = g.unpack[c][s as usize][o as usize];
                assert_eq!(g.slot[c][back as usize], s);
                if piece_type(c) != PieceType::Center {
                    assert_eq!(back, r);
                }
            }
        }
    }
}
