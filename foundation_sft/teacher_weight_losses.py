"""Sample-wise weighting of the existing three frozen-teacher KD objectives.

The supervised loss is deliberately external and is added exactly once by the
training entry point. All methods use the same per-teacher global projectors
and the same FP32 Bernoulli-KL, cosine and off-diagonal Gram losses.
"""

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from .distillation_losses import (
    _epsilon, _finite, _grid, _logits, _normalize, _pool_tokens,
    _same_batch_device, _temperature, _tensor,
)
from .equal_teacher_losses import EqualTeacherDistillationLoss, TEACHER_NAMES, _teacher_mapping


METHODS = ("equal", "ca", "rl")
POLICY_STATE_NAMES = ("teacher_logits_tanh", "teacher_ce_log1p", "soft_kd_log1p",
                      "global_kd_log1p", "local_kd_log1p", "epoch_progress")


class DirichletTeacherPolicy(nn.Module):
    """Small stochastic policy with independent sampling and REINFORCE updates.

    The stochastic action is one vector on the three-teacher probability
    simplex. Its true mean includes the positive concentration floor. Rewards
    are training proxies, not estimates of validation or test performance.
    """

    def __init__(self, seed=42, hidden_dim=32, concentration_scale=6.0,
                 concentration_floor=0.5, entropy_coefficient=0.001,
                 baseline_decay=0.95):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Policy seed must be in [0,2**32)")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Policy hidden_dim must be positive")
        for name, value in (("concentration_scale", concentration_scale),
                            ("concentration_floor", concentration_floor),
                            ("entropy_coefficient", entropy_coefficient)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if type(baseline_decay) not in (int, float) or not 0 <= baseline_decay < 1:
            raise ValueError("Policy baseline decay must be in [0,1)")
        self.seed = seed
        self.sampling_seed = seed ^ 0x504F4C49
        self.hidden_dim = hidden_dim
        self.concentration_scale = float(concentration_scale)
        self.concentration_floor = float(concentration_floor)
        self.entropy_coefficient = float(entropy_coefficient)
        self.baseline_decay = float(baseline_decay)
        # Construct only CPU layers inside an isolated stream. Constructing the
        # policy must never consume student/projector or augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed ^ 0x41435452)
            self.network = nn.Sequential(nn.Linear(16, hidden_dim), nn.Tanh(),
                                         nn.Linear(hidden_dim, len(TEACHER_NAMES)))
        self._sampler = torch.Generator(device="cpu").manual_seed(self.sampling_seed)
        self.register_buffer("reward_mean", torch.tensor(0.0))
        self.register_buffer("reward_second_moment", torch.tensor(1.0))
        self.register_buffer("reward_updates", torch.tensor(0, dtype=torch.int64))

    def concentrations(self, state):
        state = _tensor("policy state", state, 2).detach()
        if state.shape[1] != 16:
            raise ValueError("Teacher policy requires 16 detached state values")
        if state.device != self.network[0].weight.device:
            raise ValueError("Move the policy to the student's device")
        with torch.autocast(device_type=state.device.type, enabled=False):
            values = self.network(state)
            result = values.softmax(dim=-1) * self.concentration_scale + self.concentration_floor
        return _finite("Dirichlet concentrations", result)

    def actions(self, state, *, sample=True):
        concentration = self.concentrations(state)
        distribution = torch.distributions.Dirichlet(concentration)
        if sample:
            # Torch's Dirichlet sampler does not take a generator argument.
            # Sampling on CPU inside fork_rng isolates it from every CUDA RNG,
            # while the explicit stream remains serializable as a ByteTensor.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._sampler.get_state())
                weights = torch.distributions.Dirichlet(concentration.detach().cpu()).sample()
                self._sampler.set_state(torch.get_rng_state())
            weights = weights.to(concentration.device)
        else:
            weights = distribution.mean.detach()
        weights = _finite("RL teacher weights", weights.detach())
        log_prob = _finite("RL action log probability", distribution.log_prob(weights))
        entropy = _finite("Dirichlet policy entropy", distribution.entropy())
        return weights, log_prob, entropy

    def reinforce_loss(self, log_prob, entropy, rewards):
        """Score-function loss; no gradient path to the student or its rewards."""
        log_prob = _logits("action log probability", log_prob)
        entropy = _logits("policy entropy", entropy)
        rewards = _logits("training rewards", rewards).detach()
        _same_batch_device(log_prob, entropy)
        _same_batch_device(log_prob, rewards)
        old_mean = self.reward_mean.detach().clone()
        variance = (self.reward_second_moment.detach().clone() - old_mean.square()).clamp_min(1e-4)
        advantage = ((rewards - old_mean) / variance.sqrt()).clamp(-5, 5).detach()
        loss = -(advantage * log_prob).mean() - self.entropy_coefficient * entropy.mean()
        loss = _finite("REINFORCE policy loss", loss)
        mean = _finite("mean training reward", rewards.mean())
        second = _finite("training reward second moment", rewards.square().mean())
        # EMA statistics are checkpointed buffers. Zero batch variance and B=1
        # remain valid, unlike a division by the current batch's reward std.
        with torch.no_grad():
            self.reward_mean.mul_(self.baseline_decay).add_(mean * (1 - self.baseline_decay))
            self.reward_second_moment.mul_(self.baseline_decay).add_(second * (1 - self.baseline_decay))
            self.reward_updates.add_(1)
        return loss

    def stochastic_state_dict(self):
        return {"format_version": 1, "seed": self.sampling_seed,
                "cpu": self._sampler.get_state().cpu().clone()}

    def load_stochastic_state_dict(self, state):
        if (not isinstance(state, Mapping) or set(state) != {"format_version", "seed", "cpu"}
                or type(state["format_version"]) is not int or state["format_version"] != 1
                or type(state["seed"]) is not int or state["seed"] != self.sampling_seed):
            raise ValueError("Invalid policy sampling state format or seed")
        value = state["cpu"]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.device.type != "cpu" or value.layout != torch.strided
                or value.ndim != 1 or value.numel() == 0):
            raise ValueError("Policy sampling state must be a nonempty CPU ByteTensor")
        probe = torch.Generator(device="cpu")
        try:
            probe.set_state(value)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError("Policy sampling state cannot be restored by this runtime") from error
        if probe.initial_seed() != self.sampling_seed:
            raise ValueError("Policy sampling state belongs to a different seed")
        self._sampler.set_state(probe.get_state())

    def metadata(self):
        return {"algorithm": "Dirichlet_REINFORCE_training_proxy_v1", "hidden_dim": self.hidden_dim,
            "state_dim": 16, "state_blocks": list(POLICY_STATE_NAMES),
            "concentration_scale": self.concentration_scale,
            "concentration_floor": self.concentration_floor,
            "entropy_coefficient": self.entropy_coefficient,
            "baseline_decay": self.baseline_decay, "seed": self.seed,
            "sampling_seed": self.sampling_seed, "sampler": "isolated_cpu_torch_generator",
            "reward": "negative per-sample true-label Focal plus weighted three-term KD",
            "reward_is_training_proxy": True, "uses_validation_or_test_reward": False,
            "adaptation": "MTKD-RL-inspired; not original two-head or original policy update"}


def _labels(labels, reference):
    labels = _logits("labels", labels)
    _same_batch_device(reference, labels)
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Teacher confidence requires binary training labels")
    return labels


def binary_kl_per_sample(student_logits, teacher_logits, temperature=2.0):
    """Return [B] KL(teacher || student), retaining the legacy T-squared scale."""
    temperature = _temperature(temperature)
    student = _logits("student_logits", student_logits)
    teacher = _logits("teacher_logits", teacher_logits).detach()
    _same_batch_device(student, teacher)
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = _finite("scaled student logits", student / temperature)
        teacher = _finite("scaled teacher logits", teacher / temperature)
        student_pair = torch.stack((torch.zeros_like(student), student), dim=-1)
        teacher_pair = torch.stack((torch.zeros_like(teacher), teacher), dim=-1)
        result = F.kl_div(F.log_softmax(student_pair, dim=-1),
            F.log_softmax(teacher_pair, dim=-1), reduction="none", log_target=True).sum(-1)
        result = result * temperature**2
    return _finite("per-sample binary KL", result)


def global_cosine_per_sample(projected_student, teacher_global, eps=1e-6):
    """Return [B] cosine distances in one teacher's own representation space."""
    eps = _epsilon(eps)
    student = _tensor("projected_student", projected_student, 2)
    teacher = _tensor("teacher_global", teacher_global, 2).detach()
    _same_batch_device(student, teacher)
    if student.shape[1] != teacher.shape[1]:
        raise ValueError("Projected student and teacher dimensions differ")
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = _normalize("projected_student", student, eps)
        teacher = _normalize("teacher_global", teacher, eps)
        result = 1 - (student * teacher).sum(-1).clamp(-1, 1)
    return _finite("per-sample global cosine", result)


def local_relation_per_sample(student_patches, teacher_patches, student_grid,
                              teacher_grid, pool_grid=(7, 7), eps=1e-6):
    """Return [B] off-diagonal Gram MSE without mixing separate patients."""
    eps = _epsilon(eps)
    student_grid = _grid("student_grid", student_grid)
    teacher_grid = _grid("teacher_grid", teacher_grid)
    pool_grid = _grid("pool_grid", pool_grid)
    count = pool_grid[0] * pool_grid[1]
    if count < 2:
        raise ValueError("Local relations require at least two pooled positions")
    student = _tensor("student_patches", student_patches, 3)
    teacher = _tensor("teacher_patches", teacher_patches, 3).detach()
    _same_batch_device(student, teacher)
    with torch.autocast(device_type=student.device.type, enabled=False):
        student = _pool_tokens(student, student_grid, pool_grid, "student")
        teacher = _pool_tokens(teacher, teacher_grid, pool_grid, "teacher")
        student = _normalize("student pooled tokens", student, eps)
        teacher = _normalize("teacher pooled tokens", teacher, eps)
        error = (torch.bmm(student, student.transpose(1, 2))
                 - torch.bmm(teacher, teacher.transpose(1, 2))).square()
        mask = ~torch.eye(count, dtype=torch.bool, device=student.device)
        result = error[:, mask].mean(-1)
    return _finite("per-sample local relation", result)


def teacher_cross_entropies(teacher_logits, labels):
    """Use raw binary logits at T=1, separately from KD's softening temperature."""
    logits = _tensor("teacher_logits", teacher_logits, 2).detach()
    labels = _labels(labels, logits)
    return _finite("teacher training CE", F.binary_cross_entropy_with_logits(
        logits, labels[:, None].expand_as(logits), reduction="none"))


def confidence_weights(teacher_logits, labels):
    """CA-MKD response weighting: (1-softmax(teacher CE))/(K-1).

    This is the original response-weight rule, not entropy weighting or
    softmax(-CE). Its extension to all three KD terms is a task adaptation.
    """
    errors = teacher_cross_entropies(teacher_logits, labels)
    if errors.shape[1] < 2:
        raise ValueError("CA weighting requires at least two teachers")
    return _finite("CA teacher weights", (1 - errors.softmax(dim=1)) / (errors.shape[1] - 1))


class WeightedTeacherDistillationLoss(EqualTeacherDistillationLoss):
    """Apply one [B,3] teacher-weight vector to each of the three KD terms.

    Epochs are one-based. Every method uses 1/3 during the first epoch, so
    neither warm-up nor weighting changes the total scale of the KD objective.
    """

    def __init__(self, teacher_dims, student_dim=192, method="equal", seed=42,
                 temperature=2.0, soft_weight=0.5, global_weight=0.1,
                 local_weight=1.0, pool_grid=(7, 7), eps=1e-6):
        if method not in METHODS:
            raise ValueError("Teacher-weight method must be equal, ca or rl")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Teacher-weight seed must be in [0, 2**32)")
        super().__init__(teacher_dims, student_dim, temperature, soft_weight,
                         global_weight, local_weight, pool_grid, eps)
        self.method, self.seed = method, seed
        self.policy = DirichletTeacherPolicy(seed=seed) if method == "rl" else None

    def projector_parameters(self):
        return self.criteria.parameters()

    def policy_parameters(self):
        return iter(()) if self.policy is None else self.policy.parameters()

    def projector_state_dict(self):
        return {key: value.detach().cpu().clone() for key, value in self.criteria.state_dict().items()}

    def stochastic_state_dict(self):
        return None if self.policy is None else self.policy.stochastic_state_dict()

    def load_stochastic_state_dict(self, state):
        if self.policy is not None:
            self.policy.load_stochastic_state_dict(state)
        elif state is not None:
            raise ValueError("A deterministic teacher-weight method has no stochastic state")

    def weighting_metadata(self):
        value = {"method": self.method, "teachers": list(TEACHER_NAMES),
            "one_weight_vector_for_all_kd_terms": True, "warmup_epochs": 1,
            "temperature": self.temperature, "soft_weight": self.soft_weight,
            "global_weight": self.global_weight, "local_weight": self.local_weight,
            "pool_grid": list(self.pool_grid), "eps": self.eps}
        if self.method == "ca":
            value["weight_rule"] = "(1-softmax_teacher_CE)/(K-1), raw logits at T=1"
            value["adaptation"] = "CA-MKD response weighting applied to three KD terms"
        if self.policy is not None:
            value["policy"] = self.policy.metadata()
        return value

    def _policy_state(self, teacher_logits, labels, terms, epoch, total_epochs):
        errors = teacher_cross_entropies(teacher_logits, labels)
        blocks = [torch.tanh(teacher_logits.detach() / 10), errors.detach().clamp_min(0).log1p()]
        blocks.extend(terms[key].detach().clamp_min(0).log1p() for key in ("soft", "global", "local"))
        progress = teacher_logits.new_full((len(teacher_logits), 1), epoch / total_epochs)
        return _finite("detached teacher policy state", torch.cat([*blocks, progress], dim=1)).detach()

    def _per_sample(self, student_logits, references, student_global,
                    student_patches, student_grid):
        logits = _logits("student_logits", student_logits)
        _teacher_mapping(references, "references")
        terms = {name: [] for name in ("soft", "global", "local")}
        raw_teacher_logits = []
        for name in TEACHER_NAMES:
            reference = references[name]
            if not isinstance(reference, Mapping):
                raise ValueError("Each teacher reference must be a feature mapping")
            teacher_logits = _logits(name + " logits", reference.get("logits")).detach()
            _same_batch_device(logits, teacher_logits)
            raw_teacher_logits.append(teacher_logits)
            soft = global_term = local = logits.new_zeros(logits.shape)
            if self.soft_weight > 0:
                soft = binary_kl_per_sample(logits, teacher_logits, self.temperature)
            if self.global_weight > 0:
                student = _tensor("student_global", student_global, 2)
                teacher = _tensor(name + " global_features", reference.get("global_features"), 2)
                _same_batch_device(logits, student)
                _same_batch_device(student, teacher)
                if student.shape[1] != self.student_dim or teacher.shape[1] != self.teacher_dims[name]:
                    raise ValueError("Unexpected global feature dimensions")
                projector = self.criteria[name].projector
                if projector.weight.device != student.device:
                    raise ValueError("Move the KD module to the student's device")
                with torch.autocast(device_type=student.device.type, enabled=False):
                    projected = F.linear(student, projector.weight.float())
                global_term = global_cosine_per_sample(projected, teacher, self.eps)
            if self.local_weight > 0:
                student = _tensor("student_patches", student_patches, 3)
                _same_batch_device(logits, student)
                local = local_relation_per_sample(student, reference.get("local_features"),
                    student_grid, reference.get("grid_shape"), self.pool_grid, self.eps)
            for key, value in (("soft", soft), ("global", global_term), ("local", local)):
                terms[key].append(value)
        terms = {name: torch.stack(values, dim=1) for name, values in terms.items()}
        terms["total"] = (self.soft_weight * terms["soft"] + self.global_weight * terms["global"]
                          + self.local_weight * terms["local"])
        return logits, torch.stack(raw_teacher_logits, dim=1), terms

    def forward(self, student_logits, references, labels, *, student_global=None,
                student_patches=None, student_grid=None, epoch=1,
                total_epochs=100, hard_losses=None):
        if type(epoch) is not int or epoch < 1 or type(total_epochs) is not int or total_epochs < epoch:
            raise ValueError("Epochs must be one-based integers within the training budget")
        logits, teacher_logits, terms = self._per_sample(
            student_logits, references, student_global, student_patches, student_grid)
        _labels(labels, logits)
        weights = logits.new_full((len(logits), len(TEACHER_NAMES)), 1 / len(TEACHER_NAMES))
        policy_loss = None
        reward = policy_entropy = logits.new_zeros(())
        if self.method == "ca" and epoch > 1:
            weights = confidence_weights(teacher_logits, labels)
        if self.method == "rl" and epoch > 1:
            state = self._policy_state(teacher_logits, labels, terms, epoch, total_epochs)
            weights, log_prob, entropy = self.policy.actions(state, sample=self.training)
            policy_entropy = entropy.detach().mean()
            if hard_losses is None:
                if self.training:
                    raise ValueError("RL training needs per-sample true-label Focal losses")
            else:
                hard = _logits("per-sample hard losses", hard_losses).detach()
                _same_batch_device(logits, hard)
                if bool((hard < 0).any()):
                    raise ValueError("Per-sample true-label losses must be nonnegative")
                rewards = -(hard + (weights * terms["total"].detach()).sum(1)).detach()
                reward = _finite("mean RL training proxy", rewards.mean())
                if self.training:
                    policy_loss = self.policy.reinforce_loss(log_prob, entropy, rewards)
        result = {name: _finite("weighted " + name, (weights * values).sum(1).mean())
                  for name, values in terms.items()}
        result.update(per_teacher={name: {key: value[:, index].mean() for key, value in terms.items()}
                                   for index, name in enumerate(TEACHER_NAMES)},
            per_sample=terms, weights=weights.detach(), policy_loss=None,
            reward=reward, policy_entropy=policy_entropy,
            weight_entropy=-(weights.clamp_min(1e-12).log() * weights).sum(1).mean())
        result["policy_loss"] = policy_loss
        return result
