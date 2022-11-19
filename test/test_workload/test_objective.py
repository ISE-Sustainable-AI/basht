
class TestObjective:

    def test_objective(self, prepared_objective):
        # setup
        objective = prepared_objective

        # tesst
        objective.load()
        objective.train()

        # assert
        assert objective
        assert isinstance(objective.workload_definition, dict)
        assert objective._functional_objective
