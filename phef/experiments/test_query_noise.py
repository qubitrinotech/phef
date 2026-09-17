"""Проверки корректности возмущения query key для KT-1."""
import sys, torch
sys.path.insert(0, 'phef')
from phef.query_noise import key_scale_from_codebook, perturb_query_keys


def test_memory_untouched():
    cb = torch.randn(2, 512, 32)
    before = cb.clone()
    qk = cb[0, 5].clone().unsqueeze(0)
    perturb_query_keys(qk, 0.1, key_scale_from_codebook(cb))
    assert torch.equal(cb, before), "codebook изменён -- KT-1 измерял бы порчу памяти"


def test_sigma_zero_is_identity():
    cb = torch.randn(2, 512, 32)
    qk = cb[0, 5].clone().unsqueeze(0)
    out = perturb_query_keys(qk, 0.0, key_scale_from_codebook(cb))
    assert torch.equal(out, qk), "sigma=0 обязана быть тождественной"


def test_noise_magnitude_matches_theory():
    torch.manual_seed(0)
    cb = torch.randn(2, 512, 32)
    sc = key_scale_from_codebook(cb)
    qk = cb[0, 5].clone().unsqueeze(0)
    norms = torch.tensor([
        (perturb_query_keys(qk, 0.1, sc, torch.Generator().manual_seed(i)) - qk).norm()
        for i in range(2000)
    ])
    theory = 0.1 * sc * (32 ** 0.5)
    ratio = norms.mean().item() / theory
    assert 0.97 < ratio < 1.03, f"нормировка смещена: ratio={ratio:.4f}"


def test_key_scale_unit_for_randn():
    sc = key_scale_from_codebook(torch.randn(4, 1024, 32))
    assert 0.97 < sc < 1.03, sc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name}: OK")
