def test_imports():
    import torch, scanpy, squidpy, toponetx, topomodelx, gudhi, scipy
    import hodgecc
    assert not torch.cuda.is_available() or True