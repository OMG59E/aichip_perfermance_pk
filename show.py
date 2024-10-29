import os
import io
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

edgex_column_defs = [
    {"field": "ID", "resizable": True, "filter": True, "width": 70},
    {"field": "Desc", "resizable": True, "filter": True},
    {"field": "Model", "resizable": True, "filter": True},
    {"field": "MD5", "resizable": True, "filter": True},
    {"field": "Toolkit", "resizable": True, "filter": True, "width": 100},
    {"field": "Target", "resizable": True, "filter": True, "width": 100},
    {"field": "Cube", "resizable": True, "filter": True, "width": 100},
    {"field": "Opt", "resizable": True, "filter": True, "width": 100},
    {"field": "MSG", "resizable": True, "filter": True, "cellRenderer": "agAnimateShowChangeCellRenderer"},
    {"field": "Time", "resizable": True, "filter": True},
    {"field": "QuantizationSpan", "resizable": True, "filter": True, "width": 100},
    {"field": "BuildSpan", "resizable": True, "filter": True, "width": 100},
    {"field": "Latency", "resizable": True, "filter": True, "width": 100},
    {"field": "CostModel", "resizable": True, "filter": True, "width": 100},
    {"field": "Throughput", "resizable": True, "filter": True, "width": 100},
]
    

def fetch_edgex_data():
    query = "SELECT * FROM edgex"
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
        new_record["Cube"] = record["num_cube"]
        new_record["Opt"] = record["opt_level"]
        new_record["MSG"] = record["msg"] if len(record["msg"]) <= 8192 else record["msg"][-8192:]
        new_record["Time"] = str(record["build_time"])
        new_record["QuantizationSpan"] = record["quantization_span"]
        new_record["BuildSpan"] = record["build_span"]
        latency = record["latency"]
        costmodel = record["costmodel_latency"]
        if latency != 0:
            new_record["Latency"] = f"{latency:.3f}"
            new_record["CostModel"] = f"{costmodel:.3f}"
            new_record["Throughput"] = f"{1000 / latency:.2f}"
        new_records.append(new_record)
    return new_records


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
    dcc.Tabs(
        id="current-table-tabs",
        value="edgex",
        style={"height": "50px", "fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},  # 控制标签头的整体高度和字体大小
        children=[
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="EDGEX", 
                value="edgex",
                children=[
                    dag.AgGrid(
                        id="edgex-grid",
                        columnDefs=edgex_column_defs,
                        rowData=fetch_edgex_data(),
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
                        csvExportParams={"filename": "edgex.csv"},
                    ),
                    dcc.Interval(
                        id="update_edgex",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    ),
                    # 添加下载按钮
                    dcc.Download(id="edgex-download"),  # 添加 dcc.Download 组件
                    html.Div(
                        dbc.Button("Export CSV", id="edgex-download-button", color="primary", className="mr-1"),
                        style={"display": "flex", "justify-content": "center", "padding": "10px"},
                    ),
                ]
            ),
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="AXERA",
                value="axera",
                children=[
                    dag.AgGrid(
                        id="axera-grid",
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
                        csvExportParams={"filename": "axera.csv"},
                    ),
                    dcc.Interval(
                        id="update_axera",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    ),
                    # 添加下载按钮
                    dcc.Download(id="axera-download"),  # 添加 dcc.Download 组件
                    html.Div(
                        dbc.Button("Export CSV", id="axera-download-button", color="primary", className="mr-1"),
                        style={"display": "flex", "justify-content": "center", "padding": "10px"},
                    ),
                ]
            ),
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="SOPHGO",
                value="sophgo",
                children=[
                    dag.AgGrid(
                        id="sophgo-grid",
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
                        csvExportParams={"filename": "sophgo.csv"},
                    ),
                    dcc.Interval(
                        id="update_sophgo",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    ),
                    # 添加下载按钮
                    dcc.Download(id="sophgo-download"),  # 添加 dcc.Download 组件
                    html.Div(
                        dbc.Button("Export CSV", id="sophgo-download-button", color="primary", className="mr-1"),
                        style={"display": "flex", "justify-content": "center", "padding": "10px"},
                    ),
                ]
            ),
            dcc.Tab(
                style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                selected_style={"fontSize": "14px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "bold"},
                label="Rockchip",
                value="rockchip",
                children=[
                    dag.AgGrid(
                        id="rockchip-grid",
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
                        csvExportParams={"filename": "rockchip.csv"},
                    ),
                    dcc.Interval(
                        id="update_rockchip",
                        interval=30*1000,  # 以毫秒为单位，60秒
                        n_intervals=0
                    ),
                    # 添加下载按钮
                    dcc.Download(id="rockchip-download"),  # 添加 dcc.Download 组件
                    html.Div(
                        dbc.Button("Export CSV", id="rockchip-download-button", color="primary", className="mr-1"),
                        style={"display": "flex", "justify-content": "center", "padding": "10px"},
                    ),
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
    Output("edgex-grid", "exportDataAsCsv"),
    Input("edgex-download-button", "n_clicks"),
)
def export_edgex_data_as_csv(n_clicks):
    if n_clicks:
        return True
    return False


@app.callback(
    Output("axera-grid", "exportDataAsCsv"),
    Input("axera-download-button", "n_clicks"),
)
def export_axera_data_as_csv(n_clicks):
    if n_clicks:
        return True
    return False


@app.callback(
    Output("sophgo-grid", "exportDataAsCsv"),
    Input("sophgo-download-button", "n_clicks"),
)
def export_sophgo_data_as_csv(n_clicks):
    if n_clicks:
        return True
    return False


@app.callback(
    Output("rockchip-grid", "exportDataAsCsv"),
    Input("rockchip-download-button", "n_clicks"),
)
def export_rockchip_data_as_csv(n_clicks):
    if n_clicks:
        return True
    return False


@app.callback(
    Output("download-excel", "data"),
    Input("download-button", "n_clicks"),
    State("axera-grid", "rowData"),  # 替换成你要下载的表格数据
    State("sophgo-grid", "rowData"),
    State("rockchip-grid", "rowData"),
    State("edgex-grid", "rowData"),
    State("current-table-tabs", "value"),  # 传入当前选中的表格名
    prevent_initial_call=True
)
def download_filtered_data(n_clicks, axera_data, sophgo_data, rockchip_data, edgex_data, current_table_tabs):
    # 检查哪个表格当前选中，假设只下载当前选中的数据
    if axera_data and current_table_tabs == "axera":
        data = pd.DataFrame(axera_data)
    elif sophgo_data and current_table_tabs == "sophgo":
        data = pd.DataFrame(sophgo_data)
    elif rockchip_data and current_table_tabs == "rockchip":
        data = pd.DataFrame(rockchip_data)
    elif edgex_data and current_table_tabs == "edgex":
        data = pd.DataFrame(edgex_data)
    else:
        return dash.no_update
    
    # 将 DataFrame 转换为 Excel 文件
    output_stream = io.BytesIO()
    data.to_excel(output_stream, index=False)
    output_stream.seek(0)
    
    # 返回下载文件
    return dcc.send_bytes(output_stream.getvalue(), f"{current_table_tabs}.xlsx")


@app.callback(
    Output("modal", "is_open"),
    Output("modal-body", "children"),
    Input("axera", "cellClicked"),
    Input("sophgo", "cellClicked"),
    Input("rockchip", "cellClicked"),
    Input("edgex", "cellClicked"),
    Input("close", "n_clicks"),
    State("modal", "is_open"),
)
def display_message(cell1, cell2, cell3, cell4, n_clicks, is_open):
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
    if trigger_id == "edgex":
        cell = cell4
    if (trigger_id in ["axera", "sophgo", "rockchip", "edgex"] and cell and cell["colId"] == "MSG"):
        msg_content = cell["value"]
        return True, html.Pre(msg_content, style={"whiteSpace": "pre-wrap", "fontSize": "14px"})
    if trigger_id == "close" and n_clicks:
        return False, ""
    return is_open, ""
    

@app.callback(
    Output("edgex", "rowData"),
    [Input("update_edgex", "n_intervals")]  # 添加分页输入
)
def update_edgex(n_intervals):
    return fetch_edgex_data()


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
    app.run_server(host="0.0.0.0", port=8091, debug=False)