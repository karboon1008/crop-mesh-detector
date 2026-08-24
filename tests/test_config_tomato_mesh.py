from __future__ import annotations

from src.config import Config


def test_config_yaml_has_tomato_mesh_block():
    cfg = Config.load()  # repo-root config.yaml
    assert cfg.get("tomato_mesh.crop") == "Tomato"
    assert cfg.get("tomato_mesh.num_nodes") == 3
    assert cfg.get("tomato_mesh.dirichlet_alpha") == 0.3
    assert cfg.get("tomato_mesh.test_fraction") == 0.20
    assert cfg.get("tomato_mesh.dedup_threshold") == 5
    assert cfg.get("tomato_mesh.dedup_max_group_size") == 25
    assert cfg.get("tomato_mesh.rounds") == 5
    assert cfg.get("tomato_mesh.plantdoc_root") == "data/PlantDoc"
    assert cfg.get("tomato_mesh.plantwild_v1_root") == "data/PlantWild/plantwild/plantwild/images"
    assert cfg.get("tomato_mesh.plantwild_v2_root") == "data/PlantWild/plantwild_v2/plantwild_v2"
