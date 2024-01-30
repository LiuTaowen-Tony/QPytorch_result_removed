
import altair as alt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Callable, Dict



def instrument(module: nn.Module) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    instrument_recursive(module, stats)
    return stats


def instrument_recursive(
    module: nn.Module, stats: Dict[str, Dict[str, float]], name: str = ""
) -> None:
    children = list(module.named_children())
    if children:
        for c_name, c in children:
            _name = f"{name}.{c_name}" if name and name != "blocks" else c_name
            instrument_recursive(c, stats, _name)
    else:
        instrument_terminal(module, stats, name)


def instrument_terminal(
    module: nn.Module, stats: Dict[str, Dict[str, float]], name: str = ""
) -> None:
    module_stats: Dict[str, float] = {}

    def require_input_grads(_module: nn.Module, input: Any) -> None:
        for i in input:
            if isinstance(i, Tensor) and i.is_floating_point():
                i.requires_grad_()

    module.register_forward_pre_hook(require_input_grads)

    if name.split(".")[-1] == "softmax":
        return

    def record_fwd_scale(_module: nn.Module, input: Any, output: Any) -> None:
        if isinstance(output, Tensor) and output.is_floating_point():
            module_stats["x"] = np.log2(output.std().item())

    module.register_forward_hook(record_fwd_scale)

    def record_bwd_scales(
        _module: nn.Module, grad_input: Any, grad_output: Any
    ) -> None:
        grad_input = list(grad_input)
        for g in grad_input:
            if (
                g is not None
                and isinstance(g, Tensor)
                and g.is_floating_point()
                and len(grad_input) == 1
            ):
                module_stats["grad_x"] = np.log2(g.std().item())

        for param_name, param in _module.named_parameters():
            if param_name == "weight":
                module_stats["w"] = np.log2(param.std().item())
                if param.grad is not None:
                    module_stats["grad_w"] = np.log2(param.grad.std().item())

    module.register_full_backward_hook(record_bwd_scales)

    stats[name] = module_stats


def visualise(stats: Dict[str, Dict[str, float]], loss, subnormal: bool = False,dir="test.png") -> None:
    df = pd.DataFrame(stats)
    df = df.stack().to_frame("scale (log₂)").reset_index(names=["type", "op"])
    chart = plot(df, loss, subnormal)
    chart.save(dir)


def plot(df: pd.DataFrame,loss, subnormal: bool = False):
    is_x_or_grad_x = (df["type"] == "x") | (df["type"] == "grad_x")
    op_order = df[df["type"] == "x"]["op"].tolist()
    colors = ["#6C8EBF", "#FF8000", "#5D8944", "#ED3434"]
    x_range = np.arange(-18 if subnormal else -14, 18 + 1 if subnormal else 16 + 1, 2)

    fp16_min = alt.Chart().mark_rule(strokeDash=(4, 4)).encode(x=alt.datum(-14))
    fp16_min_text = (
        alt.Chart()
        .mark_text(dy=-740)
        .encode(text=alt.Text(value="Min FP16 (normal)"), x=alt.datum(-10))
    )
    fp16_max = alt.Chart().mark_rule(strokeDash=(4, 4)).encode(x=alt.datum(16))
    fp16_max_text = (
        alt.Chart()
        .mark_text(dy=-740)
        .encode(text=alt.Text(value="Max FP16"), x=alt.datum(13))
    )

    x_chart = (
        alt.Chart(df[is_x_or_grad_x])
        .mark_line()
        .encode(
            x=alt.X(
                f"scale (log₂) loss({loss}):Q",
                axis=alt.Axis(orient="top", values=x_range),
                scale=alt.Scale(domain=[x_range[0], x_range[-1]]),
            ),
            y=alt.Y("op:O", title="", sort=op_order),
            color=alt.Color(
                "type",
                legend=alt.Legend(title="", labelFontSize=12, symbolSize=100),
                scale=alt.Scale(range=colors[:2]),
                sort="descending",
            ),
        )
    )
    w_chart = (
        alt.Chart(df[~is_x_or_grad_x])
        .mark_point(size=100)
        .encode(
            x=alt.X(
                "scale (log₂):Q",
                axis=alt.Axis(orient="top", values=x_range),
                scale=alt.Scale(domain=[x_range[0], x_range[-1]]),
            ),
            y=alt.Y("op:O", title="", sort=op_order),
            color=alt.Color(
                "type",
                legend=alt.Legend(title="", labelFontSize=12, symbolSize=100),
                scale=alt.Scale(range=colors[2:]),
                sort="descending",
            ),
            shape=alt.Shape(
                "type",
                scale=alt.Scale(range=["square", "triangle-down"]),
                sort="descending",
            ),
        )
    )
    layers = [x_chart, w_chart]
    if subnormal:
        layers += [fp16_min, fp16_max, fp16_min_text, fp16_max_text]
    combined_chart = (
        alt.layer(*layers)
        .resolve_scale(color="independent", shape="independent")
        .configure_axis(labelFontSize=12, titleFontSize=16)
        .properties(width=500)
    )
    return combined_chart


def analyse_full_model(
    model: nn.Module, batch_size: int = 64, seq_len: int = 16
) -> None:
    stats = instrument(model)
    y = torch.sum(model(torch.randn((1, 3, 32, 32))))
    y.backward()
    visualise(stats, subnormal=True)
