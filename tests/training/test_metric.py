from typing import Tuple, List

import pytest
import torch

from pydgn.training.callback.metric import Metric, AdditiveLoss, MultiScore
from pydgn.training.event.state import State


class FakeMetric(Metric):
    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
        )
        self.called = 0
        self.num_nodes = 20

    @property
    def name(self) -> str:
        return "Fake Metric"

    def get_predictions_and_targets(
        self, targets: torch.Tensor, *outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = torch.arange(self.num_nodes).float() + float(self.called)
        return pred, torch.zeros(1)

    def compute_metric(
        self, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.tensor:
        return predictions.mean()


@pytest.fixture
def fake_metric():
    def metric_init_fun(
        use_as_loss, reduction, accumulate_over_epoch, force_cpu
    ):
        return FakeMetric(
            use_as_loss, reduction, accumulate_over_epoch, force_cpu
        )

    # Return how many times the fake metric will be summed over
    return metric_init_fun


def test_metric(fake_metric):
    """
    Check that batch/epoch loss/scores are correctly computed in terms of
    averaging over batches or over the entire loss
    """
    # repetition serves to deal with additive loss, where the same mocked
    # loss will be summed over many times

    reduction = "mean"  # not necessary in this test
    for use_as_loss in [False, True]:
        for accumulate_over_epoch in [False, True]:
            for force_cpu in [False, True]:
                metric = fake_metric(
                    use_as_loss, reduction, accumulate_over_epoch, force_cpu
                )
                for num_batch_calls in [1, 10]:
                    for ep_start, ba_start, forw, comp_met, ba_end, ep_end in [
                        (
                            metric.on_training_epoch_start,
                            metric.on_training_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_training_batch_end,
                            metric.on_training_epoch_end,
                        ),
                        (
                            metric.on_eval_epoch_start,
                            metric.on_eval_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_eval_batch_end,
                            metric.on_eval_epoch_end,
                        ),
                    ]:
                        # counter used to make the score change a bit
                        metric.called = 0.0

                        num_nodes = metric.num_nodes
                        state = State(model=None, optimizer=None, device="cpu")
                        state.batch_outputs = (torch.ones(1), torch.ones(1))
                        state.batch_targets = (torch.ones(1), torch.ones(1))

                        # Simulate training epoch
                        ep_start(state)

                        for batch in range(num_batch_calls):
                            state.update(batch_loss=None)
                            state.update(batch_score=None)

                            ba_start(state)

                            forw(state)

                            comp_met(state)

                            ba_end(state)

                            # change a bit the next score
                            metric.called += 1

                        ep_end(state)

                        expected_results = [
                            torch.arange(num_nodes) + float(i)
                            for i in range(num_batch_calls)
                        ]
                        if accumulate_over_epoch:
                            expected_results = torch.cat(expected_results)
                        else:
                            # for each batch compute the average score and then
                            # average again
                            expected_results = torch.stack(
                                expected_results,
                                dim=0,
                            )
                            # average resuls for each individual batch
                            expected_results = expected_results.mean(dim=1)

                        assert (
                            expected_results.mean()
                            == state.epoch_loss[metric.name]
                            if use_as_loss
                            else state.epoch_score[metric.name]
                        )


class FakeAdditiveLoss(AdditiveLoss):
    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        **losses
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
            **losses
        )
        self.called = 0
        self.num_nodes = 20

    @property
    def name(self) -> str:
        return "Fake Additive Loss"


@pytest.fixture
def fake_additive_loss():
    def metric_init_fun(
        use_as_loss, reduction, accumulate_over_epoch, force_cpu
    ):
        return FakeAdditiveLoss(
            use_as_loss,
            reduction,
            accumulate_over_epoch,
            force_cpu,
            loss1="tests.training.test_metric.FakeMetric",
            loss2="tests.training.test_metric.FakeMetric",
            loss3="tests.training.test_metric.FakeMetric",
        )

    # Return how many times the fake metric will be summed over
    return metric_init_fun


def test_additive_loss(fake_additive_loss):
    """
    Check that batch/epoch loss/scores are correctly computed in terms of
    averaging over batches or over the entire loss
    """
    # repetition serves to deal with additive loss, where the same mocked
    # loss will be summed over many times

    reduction = "mean"  # not necessary in this test
    for use_as_loss in [True]:
        for accumulate_over_epoch in [False, True]:
            for force_cpu in [False, True]:
                metric = fake_additive_loss(
                    use_as_loss, reduction, accumulate_over_epoch, force_cpu
                )
                for num_batch_calls in [1, 10]:
                    for ep_start, ba_start, forw, comp_met, ba_end, ep_end in [
                        (
                            metric.on_training_epoch_start,
                            metric.on_training_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_training_batch_end,
                            metric.on_training_epoch_end,
                        ),
                        (
                            metric.on_eval_epoch_start,
                            metric.on_eval_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_eval_batch_end,
                            metric.on_eval_epoch_end,
                        ),
                    ]:
                        # counter used to make the score change a bit
                        for m in metric.losses:
                            m.called = 0.0

                        num_nodes = metric.num_nodes
                        state = State(model=None, optimizer=None, device="cpu")
                        state.batch_outputs = (torch.ones(1), torch.ones(1))
                        state.batch_targets = (torch.ones(1), torch.ones(1))

                        # Simulate training epoch
                        ep_start(state)

                        for batch in range(num_batch_calls):
                            state.update(batch_loss=None)
                            state.update(batch_score=None)

                            ba_start(state)

                            forw(state)

                            comp_met(state)

                            ba_end(state)

                            # change a bit the next score
                            for m in metric.losses:
                                m.called += 1.0

                        ep_end(state)

                        expected_results = [
                            (torch.arange(num_nodes) + float(i)) * 3.0
                            for i in range(num_batch_calls)
                        ]
                        if accumulate_over_epoch:
                            expected_results = torch.cat(expected_results)
                        else:
                            # for each batch compute the average score and then
                            # average again
                            expected_results = torch.stack(
                                expected_results,
                                dim=0,
                            )
                            # average resuls for each individual batch
                            expected_results = expected_results.mean(dim=1)

                        assert (
                            expected_results.mean()
                            == state.epoch_loss[metric.name]
                            if use_as_loss
                            else state.epoch_score[metric.name]
                        )


class FakeMultiScore(MultiScore):
    def __init__(
        self,
        use_as_loss: bool = False,
        reduction: str = "mean",
        accumulate_over_epoch: bool = True,
        force_cpu: bool = True,
        device: str = "cpu",
        main_scorer: Metric = None,
        **extra_scorers
    ):
        super().__init__(
            use_as_loss=use_as_loss,
            reduction=reduction,
            accumulate_over_epoch=accumulate_over_epoch,
            force_cpu=force_cpu,
            device=device,
            main_scorer=main_scorer,
            **extra_scorers
        )
        self.called = 0
        self.num_nodes = 20

    @property
    def name(self) -> str:
        return "Fake Multi Score"


@pytest.fixture
def fake_multi_score():
    def metric_init_fun(
        use_as_loss, reduction, accumulate_over_epoch, force_cpu
    ):
        return FakeMultiScore(
            use_as_loss,
            reduction,
            accumulate_over_epoch,
            force_cpu,
            main_scorer="tests.training.test_metric.FakeMetric",
            score2="tests.training.test_metric.FakeMetric",
            score3="tests.training.test_metric.FakeMetric",
        )

    # Return how many times the fake metric will be summed over
    return metric_init_fun


def test_multi_score(fake_multi_score):
    """
    Check that batch/epoch loss/scores are correctly computed in terms of
    averaging over batches or over the entire loss
    """
    # repetition serves to deal with additive loss, where the same mocked
    # loss will be summed over many times

    reduction = "mean"  # not necessary in this test
    for use_as_loss in [False]:
        for accumulate_over_epoch in [False, True]:
            for force_cpu in [False, True]:
                metric = fake_multi_score(
                    use_as_loss, reduction, accumulate_over_epoch, force_cpu
                )
                for num_batch_calls in [1, 10]:
                    for ep_start, ba_start, forw, comp_met, ba_end, ep_end in [
                        (
                            metric.on_training_epoch_start,
                            metric.on_training_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_training_batch_end,
                            metric.on_training_epoch_end,
                        ),
                        (
                            metric.on_eval_epoch_start,
                            metric.on_eval_batch_start,
                            metric.on_forward,
                            metric.on_compute_metrics,
                            metric.on_eval_batch_end,
                            metric.on_eval_epoch_end,
                        ),
                    ]:
                        # counter used to make the score change a bit
                        for s in metric.scores:
                            s.called = 0.0

                        num_nodes = metric.num_nodes
                        state = State(model=None, optimizer=None, device="cpu")
                        state.batch_outputs = (torch.ones(1), torch.ones(1))
                        state.batch_targets = (torch.ones(1), torch.ones(1))

                        # Simulate training epoch
                        ep_start(state)

                        for batch in range(num_batch_calls):
                            state.update(batch_loss=None)
                            state.update(batch_score=None)

                            ba_start(state)

                            forw(state)

                            comp_met(state)

                            ba_end(state)

                            # change a bit the next score
                            for s in metric.scores:
                                s.called += 1.0

                        ep_end(state)

                        expected_results = [
                            torch.arange(num_nodes) + float(i)
                            for i in range(num_batch_calls)
                        ]
                        if accumulate_over_epoch:
                            expected_results = torch.cat(expected_results)
                        else:
                            # for each batch compute the average score and then
                            # average again
                            expected_results = torch.stack(
                                expected_results,
                                dim=0,
                            )
                            # average resuls for each individual batch
                            expected_results = expected_results.mean(dim=1)

                        assert (
                            expected_results.mean()
                            == state.epoch_loss[metric.get_main_metric_name()]
                            if use_as_loss
                            else state.epoch_score[
                                metric.get_main_metric_name()
                            ]
                        )
