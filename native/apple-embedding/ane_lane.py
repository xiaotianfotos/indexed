"""Prioritize D's short recurrence burst between Core ML vision submissions."""
from contextlib import contextmanager
import threading


class ANELane:
    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        self._priority_waiters = 0

    @contextmanager
    def claim(self, priority=False):
        with self._condition:
            if priority:
                self._priority_waiters += 1
            try:
                self._condition.wait_for(lambda: not self._active and (priority or not self._priority_waiters))
                self._active = True
            finally:
                if priority:
                    self._priority_waiters -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()


def prioritize_recurrence(engine):
    runtime = getattr(engine, '_private_ane_runtime', None)
    if not runtime or runtime['recurrence'] is None:
        return
    from mlx_vlm.models.qwen3_5.gated_delta import register_qwen3_5_gated_delta_prefill_backend
    lane = ANELane()
    engine._ane_lane = lane
    recurrence = runtime['recurrence']
    selected = set(runtime['recurrence_profile']['decoder_layers'])
    def scheduled(q, k, v, g, beta, state, mask, context):
        if getattr(context, '_qwen3_5_decoder_layer_idx', None) in selected:
            with lane.claim(priority=True):
                return recurrence(q, k, v, g, beta, state, mask, context)
        return recurrence(q, k, v, g, beta, state, mask, context)
    register_qwen3_5_gated_delta_prefill_backend(scheduled)
