import pytest

import rubik


def test_cube_basics():
    c = rubik.Cube()
    assert c.is_solved()
    c.apply("R U R' U'").apply("(R U R' U')5")
    assert c.is_solved()
    assert rubik.Cube(alg="R").facelets() == "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"


def test_bandage_formats_and_legality():
    a = rubik.Cube(bandage="UF+UFR | D DF")
    b = rubik.Cube(bandage=["UF UFR", "D-DF"])
    c = rubik.Cube(bandage=[["UF", "UFR"], ["D", "DF"]])
    assert a.bandage == b.bandage == c.bandage
    assert not a.can_apply("R")
    assert a.can_apply("U D L B")
    with pytest.raises(ValueError):
        a.apply("U F")  # after U the block sits at UL+UFL, which F cuts
    assert a.is_solved()  # unchanged after the failed apply
    assert set(a.legal_moves("R U")) == {"U", "U2", "U'"}


def test_analyze():
    e = rubik.analyze("M2 U M U2 M' U M2")
    assert e["edge_cycles"] == [["UR", "UL", "UF"]]
    assert e["pure"]
    e = rubik.analyze("(R' D' R D)2 U (D' R' D R)2 U'")
    assert len(e["twisted"]) == 2


def test_alg_utils():
    assert rubik.simplify_alg("R R U U' L") == "R2 L"
    assert rubik.invert_alg("[R, U]") == "U R U' R'"
    assert rubik.alg_length("R U2 x M") == (3, 4)


def test_solver_unbandaged():
    s = rubik.Solver(depth=4)
    assert s.layer_sizes == [1, 18, 243, 3240, 43239]
    res = s.find("corner3", max_results=5)
    assert res and res[0]["length"] == 8
    for r in res:
        e = rubik.analyze(r["alg"])
        assert len(e["corner_cycles"]) == 1 and e["n_edges"] == 0


def test_solver_bandaged_results_are_legal():
    band = "UF+UFR+UR+U | DF+D"
    s = rubik.Solver(bandage=band, depth=5)
    for kind in ("edge3", "corner3"):
        for r in s.find(kind, max_results=20):
            c = rubik.Cube(bandage=band)
            c.apply(r["alg"])  # raises if illegal
            assert not c.is_solved()
            assert set(r["pieces"]).isdisjoint({"UF", "UFR", "UR", "DF"})


def test_specific_pieces_and_extra_depth():
    s = rubik.Solver(depth=4)
    res = s.find("edge3", pieces=["UF", "UR", "UB"], pure=True, extra_depth=2, max_results=3)
    assert res and all(set(r["pieces"]) == {"UF", "UR", "UB"} for r in res)
    with pytest.raises(ValueError):
        s.find("any")  # 'any' needs pieces


def test_difficulty():
    d = rubik.difficulty("URF+UR | DLF+DBL+DL | DRB+DB+BR | UF+FR+U+F | DFR+DR+D")
    assert d["complete"] and d["states"] == 792 and d["gods_number"] == 85
    assert d["states"] == d["shapes"] * d["group_order"]
    spec = rubik.random_bandage(4, seed=3)
    assert rubik.Cube(bandage=spec).bandage


def test_human_difficulty():
    a = rubik.human_difficulty("F + FD | FL + FLD | L + LD")
    b = rubik.human_difficulty("LDB + LD + LDF | RDB + RD + RDF")
    assert a["total"] > b["total"]
    flip = lambda d: next(t for c in d["classes"] if c["name"] == "free edges" for t in c["tools"] if t["kind"] == "2-flip")
    assert flip(a)["known"] == "" and flip(a)["comm_depth"] == 6   # A: no standard flip works, must be found
    assert flip(b)["known"] != ""                                  # B: a standard M-slice flip works


def test_cube4_basics():
    c = rubik.Cube(size=4)
    assert c.size == 4 and c.is_solved() and len(c.facelets()) == 96
    c.apply("R 2R' r U (2F u')3")
    assert not c.is_solved()
    c.apply(rubik.invert_alg("R 2R' r U (2F u')3", size=4))
    assert c.is_solved()
    assert rubik.Cube(alg="R", size=4).facelets()[:16] == "UUUF" * 4
    assert len(rubik.cubie_names(4)) == 56 and len(rubik.WINGS4) == 24
    with pytest.raises(ValueError):
        rubik.Cube(alg="M", size=4)  # no middle slice on a 4x4x4
    with pytest.raises(ValueError):
        rubik.Cube(size=5)


def test_cube4_bandage_and_effects():
    c = rubik.Cube(bandage="UFR+UFr+URf", size=4)
    assert not c.can_apply("R") and not c.can_apply("2R") and c.can_apply("r")
    c = rubik.Cube(bandage="UFR+UFr+URf | Dfr+Dfl", size=4)
    assert not c.can_apply("r")  # the D center block straddles r and l
    assert not c.can_apply("D 2B")  # after D that block lies along z, cut by 2B
    assert all(c.can_apply(m) for m in ("D", "2B", "2U")) and not c.can_apply("2F") and "2R" not in c.legal_moves()
    assert len(c.bonded_facelet_pairs()) == 5
    # OLL parity: one pair of wings swapped, centers look solved
    e = rubik.analyze("2R2 B2 U2 2L U2 2R' U2 2R U2 F2 2R F2 2L' B2 2R2", size=4)
    assert len(e["edge_cycles"]) == 1 and len(e["edge_cycles"][0]) == 2
    assert e["center_cycles"] == [] and e["n_corners"] == 0


def test_solver4():
    band = "UFR+UFr+UFl+UFL"
    s = rubik.Solver(bandage=band, size=4)
    assert s.size == 4 and s.depth == 4
    for kind in ("edge3", "center3"):
        res = s.find(kind, max_results=5)
        assert res and res[0]["length"] == 8, kind
        for r in res:
            c = rubik.Cube(bandage=band, size=4)
            c.apply(r["alg"])  # raises if illegal
            assert not set(r["pieces"]) & {"UFR", "UFr", "UFl", "UFL"}
    res = s.find("edge3", pieces=["URf", "ULb", "DFr"], max_results=3)
    assert all(set(r["pieces"]) == {"URf", "ULb", "DFr"} for r in res)
    with pytest.raises(ValueError):
        s.find("edge_flip")
    with pytest.raises(ValueError):
        s.distance(rubik.Cube())  # size mismatch
