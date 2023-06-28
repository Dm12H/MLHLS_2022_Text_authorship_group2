import json
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder


def draw_barplot(series: pd.Series) -> str:
    df = pd.DataFrame([series])
    dataframe = df.T.set_axis(["probs"], axis=1)
    dataframe.sort_values(by="probs", ascending=True, inplace=True)
    fig = px.bar(dataframe, x="probs", color="probs", orientation='h',
                 color_continuous_scale="viridis_r")
    fig.update_layout({
        "yaxis_title": None,
        "xaxis_visible": False,
        "yaxis_showticklabels": True,
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
    })

    fig.update(layout_coloraxis_showscale=False)

    return json.dumps(fig, cls=PlotlyJSONEncoder)
