from capstone_kfp import train


def test_limit_threads_follows_omp_num_threads(monkeypatch):
    # 파이프라인이 CPU 상한과 같은 값을 OMP_NUM_THREADS로 준다(pipeline.py).
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    assert train.limit_threads() == 2
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    assert train.limit_threads() == 1
