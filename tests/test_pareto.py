from neuroforge.evaluation.pareto import ParetoPoint, pareto_frontier


def test_dominated_candidate_is_flagged():
    points = [
        ParetoPoint("A", {"quality": 0.94, "cost": 0.018}),
        ParetoPoint("B", {"quality": 0.91, "cost": 0.006}),
        ParetoPoint("C", {"quality": 0.80, "cost": 0.020}),  # dominated by A on both axes
    ]
    directions = {"quality": "maximize", "cost": "minimize"}
    results = {r.candidate_id: r for r in pareto_frontier(points, directions)}

    assert results["A"].is_pareto_optimal
    assert results["B"].is_pareto_optimal
    assert not results["C"].is_pareto_optimal
    assert "A" in results["C"].dominated_by


def test_equal_points_do_not_dominate_each_other():
    points = [
        ParetoPoint("A", {"quality": 0.9, "cost": 0.01}),
        ParetoPoint("B", {"quality": 0.9, "cost": 0.01}),
    ]
    results = pareto_frontier(points, {"quality": "maximize", "cost": "minimize"})
    assert all(r.is_pareto_optimal for r in results)


def test_single_point_is_pareto_optimal():
    points = [ParetoPoint("only", {"quality": 0.5})]
    results = pareto_frontier(points, {"quality": "maximize"})
    assert results[0].is_pareto_optimal
