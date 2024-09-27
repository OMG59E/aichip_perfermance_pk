import os
import pandas as pd
import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
from sqlalchemy import create_engine
from itertools import zip_longest
from natsort import natsorted


DATABASE_URI = "mysql+pymysql://root:10086@127.0.0.1/demodels"
engine = create_engine(DATABASE_URI)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


axera_column_defs = [
    {"field": "ID", "resizable": True, "filter": True, "width": 70},
    {"field": "Desc", "resizable": True, "filter": True},
    {"field": "Model", "resizable": True, "filter": True},
    {"field": "MD5", "resizable": True, "filter": True},
    {"field": "Toolkit", "resizable": True, "filter": True, "width": 100},
    {"field": "Target", "resizable": True, "filter": True, "width": 100},
    {"field": "NPU Mode", "resizable": True, "filter": True, "width": 120},
    {"field": "MSG", "resizable": True, "filter": True, "cellRenderer": "agAnimateShowChangeCellRenderer"},
    {"field": "Time", "resizable": True, "filter": True},
    {"field": "Span", "resizable": True, "filter": True, "width": 100},
    {"field": "Latency", "resizable": True, "filter": True, "width": 100},
    {"field": "Throughput", "resizable": True, "filter": True, "width": 100},
]

sophgo_column_defs = [
    {"field": "ID", "resizable": True, "filter": True, "width": 70},
    {"field": "Desc", "resizable": True, "filter": True},
    {"field": "Model", "resizable": True, "filter": True},
    {"field": "MD5", "resizable": True, "filter": True},
    {"field": "Toolkit", "resizable": True, "filter": True, "width": 100},
    {"field": "Target", "resizable": True, "filter": True, "width": 100},
    {"field": "Num Core", "resizable": True, "filter": True, "width": 120},
    {"field": "MSG", "resizable": True, "filter": True, "cellRenderer": "agAnimateShowChangeCellRenderer"},
    {"field": "Time", "resizable": True, "filter": True},
    {"field": "Span", "resizable": True, "filter": True, "width": 100},
    {"field": "Latency", "resizable": True, "filter": True, "width": 100},
    {"field": "Throughput", "resizable": True, "filter": True, "width": 100},
]

rockchip_column_defs = [
    {"field": "ID", "resizable": True, "filter": True, "width": 70},
    {"field": "Desc", "resizable": True, "filter": True},
    {"field": "Model", "resizable": True, "filter": True},
    {"field": "MD5", "resizable": True, "filter": True},
    {"field": "Toolkit", "resizable": True, "filter": True, "width": 100},
    {"field": "Target", "resizable": True, "filter": True, "width": 100},
    {"field": "MSG", "resizable": True, "filter": True, "cellRenderer": "agAnimateShowChangeCellRenderer"},
    {"field": "Time", "resizable": True, "filter": True},
    {"field": "Span", "resizable": True, "filter": True, "width": 100},
    {"field": "Latency 1Core", "resizable": True, "filter": True, "width": 100},
    {"field": "Latency", "resizable": True, "filter": True, "width": 100},
    {"field": "Throughput", "resizable": True, "filter": True, "width": 100},
]
    

def fetch_axera_data():
    query = "SELECT * FROM axera"
    records = pd.read_sql_query(query, engine).to_dict("records")  # [{}, {}, ...]
    new_records = list()
    for record in records:
        new_record = {item["field"]: "" for item in axera_column_defs}
        new_record["ID"] = record["id"]
        new_record["Desc"] = record["desc"]
        new_record["Model"] = record["model_name"]
        new_record["MD5"] = record["md5_code"]
        new_record["Toolkit"] = record["version"]
        new_record["Target"] = record["target"]
        new_record["NPU Mode"] = record["npu_mode"]
        new_record["MSG"] = record["msg"]
        new_record["Time"] = str(record["build_time"])
        new_record["Span"] = record["build_span"]
        latency = record["latency"]
        if latency != 0:
            new_record["Latency"] = f"{latency:.3f}"
            new_record["Throughput"] = f"{1000 / latency:.2f}"
        new_records.append(new_record)
    return new_records


def fetch_sophgo_data():
    query = "SELECT * FROM sophgo"
    records = pd.read_sql_query(query, engine).to_dict("records")  # [{}, {}, ...]
    new_records = list()
    for record in records:
        new_record = {item["field"]: "" for item in sophgo_column_defs}
        new_record["ID"] = record["id"]
        new_record["Desc"] = record["desc"]
        new_record["Model"] = record["model_name"]
        new_record["MD5"] = record["md5_code"]
        new_record["Toolkit"] = record["version"]
        new_record["Target"] = record["target"]
        new_record["Num Core"] = record["num_core"]
        new_record["MSG"] = record["msg"]
        new_record["Time"] = str(record["build_time"])
        new_record["Span"] = record["build_span"]
        latency = record["latency"]
        if latency != 0:
            new_record["Latency"] = f"{latency:.3f}"
            new_record["Throughput"] = f"{1000 / latency:.2f}"
        new_records.append(new_record)
    return new_records


def fetch_rockchip_data():
    query = "SELECT * FROM rockchip"
    records = pd.read_sql_query(query, engine).to_dict("records")  # [{}, {}, ...]
    new_records = list()
    for record in records:
        new_record = {item["field"]: "" for item in rockchip_column_defs}
        new_record["ID"] = record["id"]
        new_record["Desc"] = record["desc"]
        new_record["Model"] = record["model_name"]
        new_record["MD5"] = record["md5_code"]
        new_record["Toolkit"] = record["version"]
        new_record["Target"] = record["target"]
        new_record["MSG"] = record["msg"]
        new_record["Time"] = str(record["build_time"])
        new_record["Span"] = record["build_span"]
        latency = record["latency"]
        latency_1core = record["latency_1core"]
        throughput = record["throughput"]
        if latency != 0:
            new_record["Latency 1Core"] = f"{latency_1core:.3f}"
            new_record["Latency"] = f"{latency:.3f}"
            new_record["Throughput"] = f"{throughput:.2f}"
        new_records.append(new_record)
    return new_records


# DataTable的布局
app.layout = html.Div([
    # html.H1("AiChip Model Data", style={"textAlign": "center", "marginTop": "20px", "fontSize": "24px"}),
    dcc.Tabs(
        style={"height": "50px", "fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},  # 控制标签头的整体高度和字体大小
        children=[
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="AXERA", 
                children=[
                    dag.AgGrid(
                        id="axera",
                        columnDefs=axera_column_defs,
                        rowData=fetch_axera_data(),
                        defaultColDef={
                            "resizable": True,
                            "flex": 1,
                            "minWidth": 20,
                            "filter": "agTextColumnFilter"  # 默认的筛选器类型
                        },
                        style={
                            "height": "85vh",  # 高度适应视窗高度
                            # "width": "100vw",   # 宽度适应视窗宽度
                        },
                    ),
                    dcc.Interval(
                        id="update_axera",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    )
                ]
            ),
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="SOPHGO", 
                children=[
                    dag.AgGrid(
                        id="sophgo",
                        columnDefs=sophgo_column_defs,
                        rowData=fetch_sophgo_data(),
                        defaultColDef={
                            "resizable": True,
                            "flex": 1,
                            "minWidth": 20,
                            "filter": "agTextColumnFilter"  # 默认的筛选器类型
                        },
                        style={
                            "height": "85vh",  # 高度适应视窗高度
                            # "width": "100vw",   # 宽度适应视窗宽度
                        },
                    ),
                    dcc.Interval(
                        id="update_sophgo",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    )
                ]
            ),
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="Rockchip", 
                children=[
                    dag.AgGrid(
                        id="rockchip",
                        columnDefs=rockchip_column_defs,
                        rowData=fetch_rockchip_data(),
                        defaultColDef={
                            "resizable": True,
                            "flex": 1,
                            "minWidth": 20,
                            "filter": "agTextColumnFilter"  # 默认的筛选器类型
                        },
                        style={
                            "height": "85vh",  # 高度适应视窗高度
                            # "width": "100vw",   # 宽度适应视窗宽度
                        },
                    ),
                    dcc.Interval(
                        id="update_rockchip",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    )
                ]
            )
        ]
    ),
    dbc.Modal(
        children=[
            dbc.ModalHeader(dbc.ModalTitle("Message Detail")),
            dbc.ModalBody(id="modal-body"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ml-auto")
            ),
        ],
        id="modal",
        size="lg",
        is_open=False,
    ),
])

@app.callback(
    Output("modal", "is_open"),
    Output("modal-body", "children"),
    Input("axera", "cellClicked"),
    Input("sophgo", "cellClicked"),
    Input("rockchip", "cellClicked"),
    Input("close", "n_clicks"),
    State("modal", "is_open"),
)
def display_message(cell1, cell2, cell3, n_clicks, is_open):
    # 检查是否点击了msg列
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, ""
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    cell = None
    if trigger_id == "axera":
        cell = cell1
    if trigger_id == "sophgo":
        cell = cell2
    if trigger_id == "rockchip":
        cell = cell3
    if (trigger_id in ["axera", "sophgo", "rockchip"] and cell and cell["colId"] == "MSG"):
        msg_content = cell["value"]
        return True, html.Pre(msg_content, style={"whiteSpace": "pre-wrap", "fontSize": "14px"})
    if trigger_id == "close" and n_clicks:
        return False, ""
    return is_open, ""
    

@app.callback(
    Output("axera", "rowData"),
    [Input("update_axera", "n_intervals")]  # 添加分页输入
)
def update_axera(n_intervals):
    return fetch_axera_data()


@app.callback(
    Output("sophgo", "rowData"),
    [Input("update_sophgo", "n_intervals")]  # 添加分页输入
)
def update_sophgo(n_intervals):
    return fetch_sophgo_data()


@app.callback(
    Output("rockchip", "rowData"),
    [Input("update_rockchip", "n_intervals")]  # 添加分页输入
)
def update_rockchip(n_intervals):
    return fetch_rockchip_data()


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8091, debug=True)