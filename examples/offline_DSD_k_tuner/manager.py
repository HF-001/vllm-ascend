import bisect
from dataclasses import dataclass
from typing import Optional

from vllm.v1.spec_decode.metrics import SpecDecodingStats


@dataclass
class DynamicSpeculativeConfig:
    """Batch-size schedule for dynamic speculative decoding."""
    num_speculative_tokens_per_batch_size: Optional[dict[str, int]] = None
    use_online_acceptance_rate: bool = False
    batch_stats: Optional[dict[int, dict[int, float]]] = None
    max_num_speculative_tokens: Optional[int] = None
    acceptance_rate_per_pos: Optional[list[float]] = None


class DynamicSpeculativeDecodingManager:
    """Manages dynamic adjustment of speculative tokens based on batch size
    and acceptance rates."""

    def __init__(
        self,
        dynamic_config: DynamicSpeculativeConfig,
        vllm_max_batch_size: int,
        vllm_num_speculative_tokens: int,
        warmup_steps: int = 100,
    ):
        self.dynamic_config = dynamic_config
        self.vllm_max_batch_size = vllm_max_batch_size
        self.vllm_num_speculative_tokens = vllm_num_speculative_tokens
        self.use_online_acceptance_rate = dynamic_config.use_online_acceptance_rate

        assert dynamic_config.batch_stats is not None, \
            "batch_stats is required for dynamic speculative decoding"
        assert dynamic_config.max_num_speculative_tokens is not None, \
            "max_num_speculative_tokens is required"
        assert dynamic_config.acceptance_rate_per_pos is not None, \
            "acceptance_rate_per_pos is required"

        self.batch_stats: dict[int, dict[int, float]] = dynamic_config.batch_stats
        self.max_num_speculative_tokens: int = dynamic_config.max_num_speculative_tokens
        self.acceptance_rate_per_pos: list[float] = dynamic_config.acceptance_rate_per_pos
        self.available_batch_sizes = sorted(dynamic_config.batch_stats.keys())

        self.steps = 0
        self.warmup_steps = warmup_steps
        self.stats = SpecDecodingStats.new(vllm_num_speculative_tokens)

        # Sanity check
        assert vllm_num_speculative_tokens <= self.max_num_speculative_tokens
        assert self.max_num_speculative_tokens == len(self.acceptance_rate_per_pos)
        assert self.max_num_speculative_tokens > 0
        assert all(0.0 <= a <= 1.0 for a in self.acceptance_rate_per_pos)
        assert 1 in self.batch_stats, f"BS 1 not found in {self.batch_stats.keys()}"
        assert vllm_max_batch_size in self.batch_stats, \
            f"max BS {vllm_max_batch_size} not found in {self.batch_stats.keys()}"

        for bs in self.available_batch_sizes:
            assert bs > 0
            assert 0 in self.batch_stats[bs], f"BS {bs} must have draft 0 stats"
            assert 1 in self.batch_stats[bs], f"BS {bs} must have draft 1 stats"
            assert sorted(self.batch_stats[bs].keys()) == list(self.batch_stats[bs].keys()), \
                f"BS {bs} draft keys must be sorted"

        self._optimal_k_table: dict[int, int] = {}
        self._sorted_bs_keys: list[int] = []
        self.update_optimal_num_speculative_tokens()

    def update_optimal_num_speculative_tokens(self) -> None:
        self._optimal_k_table = {}
        for bs in self.available_batch_sizes:
            stats = self.batch_stats[bs]
            best_k = 0
            best_goodput = 1.0 / stats.get(0, float('inf'))  # K=0: AL=1

            for k in sorted(stats.keys()):
                if k == 0:
                    continue
                itl = stats[k]
                if itl <= 0:
                    continue
                al = self._compute_accepted_length(k)
                goodput = al / itl       # https://arxiv.org/pdf/2406.14066  Goodput
                if goodput > best_goodput:
                    best_goodput = goodput
                    best_k = k

            self._optimal_k_table[bs] = best_k
        self._sorted_bs_keys = sorted(self._optimal_k_table.keys())


    def get_optimal_num_speculative_tokens(self, batch_size: int) -> int:
        """Return optimal K for a given batch size via binary search."""
        idx = bisect.bisect_right(self._sorted_bs_keys, batch_size) - 1
        if idx < 0:
            return self._optimal_k_table[self._sorted_bs_keys[0]]
        return self._optimal_k_table[self._sorted_bs_keys[idx]]


    def _compute_accepted_length(self, k: int) -> float:
        """Compute expected accepted length for given K using unconditional rates."""
        al = 1.0  # base token from target model
        for i in range(min(k, len(self.acceptance_rate_per_pos))):
            al += self.acceptance_rate_per_pos[i] 
        return al

    def step(self, batch_size: int) -> int:
        """Called per forward pass; returns optimal K for current batch."""
        self.steps += 1
        if self.should_update():
            new_rates = self.compute_acceptance_rate_per_pos()
            self.update_acceptance_rate_per_pos(new_rates)

        return self.get_optimal_num_speculative_tokens(batch_size)

    def should_update(self) -> bool:
        """Check if online acceptance rate update is due."""
        if not self.use_online_acceptance_rate:
            return False
        return self.steps > self.warmup_steps and self.steps % self.warmup_steps == 0

    def compute_acceptance_rate_per_pos(self) -> list[float]:
        """Compute acceptance rate from cumulative online stats."""
        rates = []
        for i in range(self.vllm_num_speculative_tokens):
            drafted = self.stats.num_draft_tokens_per_pos[i]
            if drafted == 0:
                rates.append(0.0)
            else:
                accepted = self.stats.num_accepted_tokens_per_pos[i]
                rates.append(accepted / drafted)
        return rates

    def update_acceptance_rate_per_pos(self, new_rates: list[float]) -> None:
        """Update acceptance rates with latest online statistics."""
        self.acceptance_rate_per_pos = new_rates
        self.update_optimal_num_speculative_tokens()