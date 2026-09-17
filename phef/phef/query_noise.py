"""Возмущение query key для KT-1. Только запрос, память не затрагивается."""
import torch


def key_scale_from_codebook(codebook):
    """Масштаб ключа: среднее std по компонентам codebook.

    sigma в KT-1 выражается в единицах этой величины.
    Абсолютные sigma непубликуемы без нормировки.
    """
    return float(codebook.float().std(dim=tuple(range(codebook.dim() - 1))).mean())


def perturb_query_keys(query_keys, sigma, scale, generator=None):
    """Аддитивный гауссов шум ТОЛЬКО на query_keys.

    Ключи памяти живут в codebook и здесь не затрагиваются -- это
    обеспечивает семантику 'неточный запрос', а не 'испорченная память'.
    """
    if sigma == 0.0:
        return query_keys
    noise = torch.randn(
        query_keys.shape,
        device=query_keys.device,
        dtype=query_keys.dtype,
        generator=generator,
    )
    return query_keys + sigma * scale * noise