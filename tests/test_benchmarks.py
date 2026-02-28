from benchmarks.benchmark_vs_r import compare_backends, run_python_benchmark


def test_run_python_benchmark_smoke():
    out = run_python_benchmark(80, 25, 4, max_iter=20, seed=123)
    assert out.backend == "python"
    assert out.n == 80
    assert out.p == 25
    assert out.L == 4
    assert out.runtime_sec >= 0.0
    assert out.n_iter >= 1


def test_compare_backends_smoke_without_r_requirement():
    out = compare_backends(60, 20, 3, max_iter=15, seed=321)
    assert "python" in out
    assert "r" in out
    assert "speedup_python_vs_r" in out
