import plotly.graph_objects as go

def enforce_monotonic(values):
    out,cur=[],None
    for v in values:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

def funnel_fig(labels, values, title=None):
    fig=go.Figure(go.Funnel(
        y=labels, x=values, textinfo="value", textposition="inside",
        texttemplate="<b>%{value}</b>", textfont=dict(size=35),
        opacity=0.95, connector={"line":{"dash":"dot","width":1}}
    ))
    fig.update_layout(
        title=title or "", margin=dict(l=10,r=10,t=48,b=10),
        height=540, template="plotly_white", separators=".,",
        uniformtext=dict(minsize=12,mode="show")
    )
    return fig
