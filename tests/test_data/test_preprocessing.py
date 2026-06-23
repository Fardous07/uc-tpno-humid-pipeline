"""
Tests for src/data/preprocessing and src/data/datasets.

Covers: CIFSanitizer, GraphConfig, DataSplitter,
        SyntheticAdsorptionDataset, collation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# ── CIF fixture ──────────────────────────────────────────

SAMPLE_CIF = """\
data_TEST_MOF
_cell_length_a 25.832
_cell_length_b 25.832
_cell_length_c 25.832
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M P1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu1 Cu 0.0 0.0 0.0
O1 O 0.1 0.1 0.1
C1 C 0.2 0.2 0.2
C2 C 0.3 0.3 0.3
O2 O 0.4 0.4 0.4
"""


@pytest.fixture
def cif_dir(tmp_path):
    d = tmp_path / "cifs"
    d.mkdir()
    (d / "TEST_MOF.cif").write_text(SAMPLE_CIF)
    return d


# ═══════════════════════════════════════════════════════════
# SANITIZE
# ═══════════════════════════════════════════════════════════

class TestReadCell:
    def test_valid_cif(self, cif_dir):
        from src.data.preprocessing.sanitize import read_cell_from_cif
        cell = read_cell_from_cif(cif_dir / "TEST_MOF.cif")
        assert cell is not None
        assert abs(cell["a"] - 25.832) < 0.01
        assert abs(cell["gamma"] - 90.0) < 0.01

    def test_bad_cif_returns_none(self, tmp_path):
        from src.data.preprocessing.sanitize import read_cell_from_cif
        (tmp_path / "bad.cif").write_text("data_X\n_cell_length_a 10\n")
        assert read_cell_from_cif(tmp_path / "bad.cif") is None


class TestCountAtoms:
    def test_count(self, cif_dir):
        from src.data.preprocessing.sanitize import count_atoms_in_cif
        assert count_atoms_in_cif(cif_dir / "TEST_MOF.cif") == 5


class TestCIFSanitizer:
    def test_valid(self, cif_dir, tmp_path):
        from src.data.preprocessing.sanitize import CIFSanitizer
        san = CIFSanitizer(min_atoms=2, max_atoms=100)
        out = tmp_path / "clean" / "TEST_MOF.cif"
        report = san.sanitize(cif_dir / "TEST_MOF.cif", out)
        assert report["valid"]
        assert report["n_atoms_raw"] == 5
        assert out.exists()

    def test_reject_min_atoms(self, cif_dir, tmp_path):
        from src.data.preprocessing.sanitize import CIFSanitizer
        san = CIFSanitizer(min_atoms=100)
        r = san.sanitize(cif_dir / "TEST_MOF.cif", tmp_path / "x.cif")
        assert not r["valid"]

    def test_reject_max_atoms(self, cif_dir, tmp_path):
        from src.data.preprocessing.sanitize import CIFSanitizer
        san = CIFSanitizer(max_atoms=2)
        r = san.sanitize(cif_dir / "TEST_MOF.cif", tmp_path / "x.cif")
        assert not r["valid"]

    def test_reject_bad_cell(self, tmp_path):
        from src.data.preprocessing.sanitize import CIFSanitizer
        bad = tmp_path / "bad.cif"
        bad.write_text("data_B\n_cell_length_a 0.5\n_cell_length_b 25\n"
                       "_cell_length_c 25\n_cell_angle_alpha 90\n"
                       "_cell_angle_beta 90\n_cell_angle_gamma 90\n")
        san = CIFSanitizer()
        assert not san.sanitize(bad, tmp_path / "o.cif")["valid"]

    def test_batch(self, cif_dir, tmp_path):
        from src.data.preprocessing.sanitize import CIFSanitizer
        san = CIFSanitizer(min_atoms=2)
        reports = san.sanitize_batch(cif_dir, tmp_path / "out")
        assert len(reports) == 1
        assert reports[0]["valid"]


# ═══════════════════════════════════════════════════════════
# GRAPH BUILDER (structure check only — ASE/PyG may not be installed)
# ═══════════════════════════════════════════════════════════

class TestGraphConfig:
    def test_defaults(self):
        from src.data.preprocessing.graph_builder import GraphConfig
        cfg = GraphConfig()
        assert cfg.cutoff == 6.0
        assert cfg.edge_features is True

    def test_custom(self):
        from src.data.preprocessing.graph_builder import GraphConfig
        cfg = GraphConfig(cutoff=8.0, max_neighbours=20)
        assert cfg.cutoff == 8.0
        assert cfg.max_neighbours == 20


# ═══════════════════════════════════════════════════════════
# SPLITTER
# ═══════════════════════════════════════════════════════════

class TestDataSplitter:
    def test_random(self):
        from src.data.datasets.splitter import DataSplitter
        ids = [f"M_{i}" for i in range(100)]
        sp = DataSplitter("random", test_size=0.1, val_size=0.1)
        tr, va, te = sp.split(ids)
        assert len(tr) + len(va) + len(te) == 100
        assert len(set(tr) & set(te)) == 0
        assert len(set(tr) & set(va)) == 0

    def test_scaffold(self):
        import pandas as pd
        from src.data.datasets.splitter import DataSplitter
        ids = [f"M_{i}" for i in range(100)]
        meta = pd.DataFrame({"mof_id": ids, "topology": [f"t{i%8}" for i in range(100)]})
        sp = DataSplitter("scaffold", test_size=0.2, val_size=0.1)
        tr, va, te = sp.split(ids, meta)
        assert len(tr) + len(va) + len(te) == 100
        train_t = set(meta.iloc[tr]["topology"])
        test_t = set(meta.iloc[te]["topology"])
        assert len(train_t & test_t) == 0  # disjoint topologies

    def test_humidity(self):
        import pandas as pd
        from src.data.datasets.splitter import DataSplitter
        ids = [f"M_{i}" for i in range(100)]
        meta = pd.DataFrame({"mof_id": ids,
                              "humidity": [0.0]*70 + [0.15]*30})
        sp = DataSplitter("humidity")
        tr, va, te = sp.split(ids, meta)
        assert len(te) == 30

    def test_save_load(self, tmp_path):
        from src.data.datasets.splitter import DataSplitter
        ids = [f"M_{i}" for i in range(50)]
        sp = DataSplitter("random")
        tr, va, te = sp.split(ids)
        p = tmp_path / "s.json"
        sp.save_splits(ids, tr, va, te, p)
        t2, v2, e2 = DataSplitter.load_splits(p)
        assert len(t2) == len(tr)

    def test_invalid(self):
        from src.data.datasets.splitter import DataSplitter
        with pytest.raises(ValueError):
            DataSplitter("bogus")

    def test_deterministic(self):
        from src.data.datasets.splitter import DataSplitter
        ids = [f"M_{i}" for i in range(80)]
        s1 = DataSplitter("random", random_state=99)
        s2 = DataSplitter("random", random_state=99)
        assert s1.split(ids) == s2.split(ids)


# ═══════════════════════════════════════════════════════════
# SYNTHETIC DATASET
# ═══════════════════════════════════════════════════════════

class TestSyntheticDataset:
    def test_creation(self):
        torch = pytest.importorskip("torch")
        from src.data.datasets.adsorption_dataset import SyntheticAdsorptionDataset
        ds = SyntheticAdsorptionDataset(n_mofs=10, n_points=8, seed=42)
        assert len(ds) == 10
        s = ds[0]
        assert s["conditions"].shape == (8, 4)
        assert s["loadings"].shape == (8, 3)
        assert s["loadings"].min() >= 0

    def test_collation_uniform(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from src.data.datasets.adsorption_dataset import SyntheticAdsorptionDataset
        ds = SyntheticAdsorptionDataset(n_mofs=8, n_points=6)
        batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=False)))
        assert batch["conditions"].shape == (4, 6, 4)
        assert batch["loadings"].shape == (4, 6, 3)
        assert batch["mask"].all()

    def test_variable_padding(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from src.data.datasets.adsorption_dataset import SyntheticAdsorptionDataset
        ds = SyntheticAdsorptionDataset(n_mofs=4, n_points=10)
        ds.samples[0]["conditions"] = ds.samples[0]["conditions"][:3]
        ds.samples[0]["loadings"] = ds.samples[0]["loadings"][:3]
        ds.samples[0]["n_points"] = 3
        batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=False)))
        assert batch["conditions"].shape[1] == 10
        assert batch["mask"][0, 2] == True
        assert batch["mask"][0, 3] == False

    def test_trainer_compatible(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from src.data.datasets.adsorption_dataset import SyntheticAdsorptionDataset
        ds = SyntheticAdsorptionDataset(n_mofs=4, n_points=5)
        batch = next(iter(ds.get_dataloader(batch_size=2, shuffle=False)))
        assert "graphs" in batch
        assert "conditions" in batch
        assert "loadings" in batch
        assert hasattr(batch["graphs"], "z")  # PyG Batch
        assert batch["conditions"].dtype == torch.float32