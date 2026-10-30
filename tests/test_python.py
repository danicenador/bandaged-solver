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
