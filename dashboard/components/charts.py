import plotly.graph_objects as go


_FONT = "Inter, system-ui, sans-serif"
_COLORS = ["#6C63FF", "#3B82F6", "#22C55E", "#F59E0B", "#EF4444",
           "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]


def theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the unified Glassmorphism chart theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color="#8B92A5", size=12),
        title=dict(text=title, font=dict(size=15, color="#EAEDF2", family=_FONT), pad=dict(b=16)),
        margin=dict(t=48, l=8, r=8, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            font=dict(size=11, color="#8B92A5"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#8B92A5", size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(color="#8B92A5", size=11)),
        hoverlabel=dict(bgcolor="#0E1220", font_size=12, font_family=_FONT, bordercolor="#2A2E40"),
        colorway=_COLORS,
    )
    return fig


def colors():
    return _COLORS
