#! /usr/bin/env python3
import glob
import subprocess
import os, sys
import json
import resource
import argparse
import shutil
import yaml
import hashlib
import sqlalchemy
import pymysql
import onnx
import traceback
import onnx_graphsurgeon as gs
from loguru import logger
from natsort import natsorted
from datetime import datetime
from fabric import Connection
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy import Column, Integer, DateTime, String, BOOLEAN, Float, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

MAX_MEMORY = 28 * 1024 * 1024 * 1024  # GB
HOST = "mysql"
USERNAME = "root"
PASSWORD = "10086"
PORT = 3306
DATABASE = "demodels"

pymysql.install_as_MySQLdb()
engine = sqlalchemy.create_engine(
    "mysql+pymysql://{}:{}@{}:{}".format(USERNAME, PASSWORD, HOST, PORT),
    encoding="utf-8",
    pool_size=100
)  # connect to server
engine.execute("CREATE SCHEMA IF NOT EXISTS `{}`;".format(DATABASE))  # create db
engine.execute("USE `{}`;".format(DATABASE))  # select new db
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base(engine)


class EdgeXTable(Base):
    __tablename__ = "edgex3"
    __table_args__ = {"extend_existing": True, "mysql_charset": "utf8"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    desc = Column(String(512), comment="任务描述")
    model_name = Column(String(512), index=True, comment="模型名")
    md5_code = Column(String(512), comment="模型md5码")
    model_path = Column(String(512), comment="模型路径")
    version = Column(String(512), comment="工具链版本")
    target = Column(String(512), comment="目标芯片")
    opt_level = Column(String(512), comment="优化等级")
    num_cube = Column(String(512), comment="")
    msg = Column(LONGTEXT, comment="编译信息")
    build_time = Column(DateTime, comment="编译时间")
    quantization_span = Column(String(512), comment="量化耗时")
    build_span = Column(String(512), comment="编译耗时")
    latency    = Column(Float, default=0, comment="1线程推理平均时延(毫秒)")
    latency_2t = Column(Float, default=0, comment="2线程推理平均时延(毫秒)")
    latency_3t = Column(Float, default=0, comment="3线程推理平均时延(毫秒)")
    latency_4t = Column(Float, default=0, comment="4线程推理平均时延(毫秒)")
    throughput    = Column(Float, default=0, comment="1线程推理吞吐")
    throughput_2t = Column(Float, default=0, comment="2线程推理吞吐")
    throughput_3t = Column(Float, default=0, comment="3线程推理吞吐")
    throughput_4t = Column(Float, default=0, comment="4线程推理吞吐")
    costmodel_latency = Column(String(512), default="0", comment="cost model时延(毫秒)")
    compiled_model_path = Column(String(512), comment="编译后模型路径")
    compiled_model_md5 = Column(String(512), comment="编译后模型MD5")


def set_memory_limit(max_mem):
    def limit():
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))
    return limit


def build():
    try:
        logger.info("python /DEngine/tyassist/tyassist.py build -c config.yml")
        build_cmd = ["python", "/DEngine/tyassist/tyassist.py", "build", "-c", "config.yml"]
        res = subprocess.run(build_cmd, capture_output=True, text=True, check=True, timeout=6*3600)  # 6小时不结束视为超时
        if res.returncode == 0:
            logger.info(f"success:\n {res.stdout}")
            return True, "ok"
        else:
            logger.error(f"failed: {res.stderr}")
            return False, str(res.stderr)
    except subprocess.TimeoutExpired as e:
        logger.error(f"exception: {e.stderr}")
        return False, str(e.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"exception: {e.stderr}")
        return False, str(e.stderr)


def run_profile():
    try:
        logger.info("python /DEngine/tyassist/tyassist.py profile -c config.yml")
        build_cmd = ["python", "/DEngine/tyassist/tyassist.py", "profile", "-c", "config.yml"]
        res = subprocess.run(build_cmd, capture_output=True, text=True, check=True, timeout=3600)  # 2小时不结束视为超时
        if res.returncode == 0:
            logger.info(f"success:\n {res.stdout}")
            return True, "ok"
        else:
            logger.error(f"failed: {res.stderr}")
            return False, str(res.stderr)
    except subprocess.TimeoutExpired as e:
        logger.error(f"exception: {e.stderr}")
        return False, str(e.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"exception: {e.stderr}")
        return False, str(e.stderr)
    

def run_latency(model_path, num_thread):
    ip_addr = "192.168.33.61"
    username = "root"
    password = "root"
    remote_work_dir = "/root/"
    key_filename = "/root/.ssh/id_rsa"
    num_samples = 1000
    model_name = os.path.basename(model_path)
    remote_model_path = os.path.join(remote_work_dir, model_name)
    try:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            logger.info(f"Upload model: {model_name} to {remote_work_dir}")
            c.put(model_path, remote_work_dir)
            run_cmd = f"./ty_run_model.sh /config/sdk.cfg {remote_model_path} {num_samples} {num_thread}"
            logger.info(run_cmd)
            result = c.run(run_cmd, hide=True, warn=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error:\n{error}"
                logger.error(msg)
                return 0, 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output:\n{output}")
            lines = output.splitlines()
            line_list = lines[-4].split(" ")
            latency = float(line_list[-3][:-3])
            throughput = float(line_list[-1])
            logger.info(f"Exit Code: {result.return_code}")

            c.run(f"rm {remote_model_path}", hide=True)
            logger.info(f"Delete model: {remote_model_path}")
            return latency, throughput, "ok"
    except Exception as e:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            c.run(f"rm {remote_model_path}", hide=True)
        msg = f"Run model exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return 0, 0, 0, msg


def is_qnn(graph: gs.Graph):
    for node in graph.nodes:
        op_type = str(node.op)
        if op_type.startswith("Quantize") or op_type.startswith("QLinear") or op_type.startswith("Dequantize"):
            return True
    return False


def gen_cfg(onnx_filename, md5_code, target, opt_level, num_cube, toolkit_version, is_transformer=False):
    try:
        config_yml = "config.yml"
        if os.path.exists(config_yml):
            return True, "ok"
        graph = gs.import_onnx(onnx.load(onnx_filename))
        input_tensors = graph.inputs
        input_infos = list()
        for input_tensor in input_tensors:
            shape = input_tensor.shape
            if not isinstance(shape[0], int) or shape[0] < 1:
                shape[0] = 1
            if len(shape) >= 2:
                for d in range(1, len(shape)):
                    if not isinstance(shape[d], int) or shape[d] < 1:
                        msg = "Not support dynamic input model, when len(shape) >= 2"
                        logger.error(msg)
                        return False, msg

            input_info = dict()
            input_info["name"] = input_tensor.name
            input_info["shape"] = shape
            input_info["dtype"] = str(input_tensor.dtype)
            input_info["layout"] = "None"
            input_info["pixel_format"] = "None"
            input_info["mean"] = None
            input_info["std"] = None
            input_info["resize_type"] = 0
            input_info["padding_value"] = 128
            input_info["padding_mode"] = 1
            input_info["data_path"] = ""
            input_infos.append(input_info)

        basename = os.path.splitext(onnx_filename)[0]
        cfg = {
            "model": {
                "framework": "onnx" if not is_qnn(graph) else "onnx-qnn",
                "weight": onnx_filename,
                "graph": "",
                "save_dir": "outputs",
                "inputs": input_infos,
                "name": f"{basename}_{target}_cube{num_cube}_{md5_code}_v{toolkit_version}"
            },
            "build": {
                "target": target,
                "enable_quant": True,
                "enable_build": True,
                "opt_level": opt_level,
                "multi_thread": None,
                "quant": {
                    "data_dir": "",
                    "prof_img_num": 1,
                    "similarity_img_num": 1,
                    "similarity_dataset": None,
                    "debug_level": -1,
                    "opt_level": 0,
                    "calib_method": "min_max",
                    "custom_preprocess_module": None,
                    "custom_preprocess_cls": None,
                    "disable_pass": [],
                    "num_cube": num_cube,
                    "config": { "collect_mem": 4 }
                }, 
                "enable_dump": 0,
            }
        }
        if is_transformer:
            cfg["build"]["config"] = { "relay.edgex.use_aggressive_layout_rewrite": True }
            cfg["build"]["quant"]["float_list"] = ["add", "multiply"]
        with open(config_yml, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        logger.info(f"Save config file to: {os.path.abspath(config_yml)}")
        return True, "ok"
    except Exception as e:
        msg = f"Failed to gen config: {e}"
        logger.error(msg)
        return False, msg


def get_md5_code(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model build tool")
    parser.add_argument("type", type=str, choices=("build", "copy"), help="Please specify a operator")
    parser.add_argument("--models", "-m", type=str, required=True, help="Please specify a onnx model dir")
    parser.add_argument("--target", "-t", type=str, required=True, choices=("nnp400",), help="Please specify a chip target")
    parser.add_argument("--opt_level", "-o", type=int, required=False, default=2, choices=(2, 0), help="Please specify a opt_level")
    parser.add_argument("--num_cube", "-c", type=int, required=False, default=3, choices=(3, 2), help="Please specify a num_cube")
    parser.add_argument("--desc", "-d", type=str, required=True, help="Please specify a task desc")
    parser.add_argument("--output", "-s", type=str, required=True if "copy" in sys.argv else False, help="Please specify a output path")
    parser.add_argument("--transformer", action="store_true")
    args = parser.parse_args()

    # 检查表是否存在
    tables = engine.execute("show tables;").fetchall()
    tables = [table_name[0] for table_name in tables]
    if EdgeXTable.__tablename__ not in tables:
        Base.metadata.create_all(engine)
        logger.info("create table success")
    
    models_dir = args.models
    target = args.target
    opt_level = args.opt_level
    num_cube = args.num_cube
    desc = args.desc
    run_type = args.type
    toolkit_version = "1.0.4"
    is_transformer = args.transformer
    logger.info(f"Toolkit Version: {toolkit_version}")
    
    work_dir = f"outputs/edgex/{target}/{toolkit_version}"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    onnx_files = glob.glob(os.path.join(models_dir, "**/*.onnx"), recursive=True)
    sorted_file_list = natsorted(onnx_files, key=os.path.basename)
    for idx, filepath in enumerate(sorted_file_list):
        if not os.path.exists(filepath):
            logger.warning(f"Not found: {filepath}")
            continue

        md5_code = get_md5_code(filepath)
        filename = os.path.basename(filepath)
        basename, ext = os.path.splitext(filename)
        model_dir = f"{work_dir}/{basename}_{md5_code}/"
        compiled_model_path = f"outputs/{target}/{basename}_{target}_cube{num_cube}_{md5_code}_v{toolkit_version}_combine_O{opt_level}_aarch64.ty"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(os.path.join(model_dir, filename)):
            shutil.copy(filepath, model_dir)
            
        # 拷贝已经编译生成的模型到某个指定目录
        if run_type == "copy":
            output_dir = args.output
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            tymodel_path = os.path.join(model_dir, compiled_model_path)
            output_path = os.path.join(output_dir, os.path.basename(compiled_model_path))
            if os.path.exists(tymodel_path) and not os.path.exists(output_path):
                shutil.copy(tymodel_path, output_path)
                logger.info(f"copy: {tymodel_path} -> {output_path}")
            continue

        # 检查该模型是不是已经入库
        res = session.query(EdgeXTable).filter(
            EdgeXTable.md5_code == md5_code,
            EdgeXTable.version == toolkit_version,
            EdgeXTable.opt_level == opt_level,
            EdgeXTable.num_cube == num_cube,
        ).first()
        if res:
            if res.model_path == filepath:
                logger.info(f"MD5: {md5_code}, Path: {filepath}")
            else:
                # 名字不一样，但是内容一样的模型，直接复制
                model = EdgeXTable()
                model.desc = desc
                model.model_name = filename
                model.model_path = filepath
                model.md5_code = md5_code
                model.version = toolkit_version
                model.build_time = res.build_time
                model.msg = res.msg
                model.target = res.target
                model.num_cube = res.num_cube
                model.opt_level = res.opt_level
                model.quantization_span = res.quantization_span
                model.build_span = res.build_span
                model.compiled_model_path = res.tymodel_path
                model.compiled_model_md5 = res.tymodel_md5
                model.latency = res.latency
                model.costmodel_latency = res.costmodel_latency
                session.add(model)
                session.commit()
            continue

        # 新模型
        model = EdgeXTable()
        model.desc = desc
        model.model_name = filename
        model.model_path = filepath
        model.md5_code = md5_code
        model.version = toolkit_version
        build_time = datetime.now()
        model.build_time = build_time
        model.num_cube = num_cube
        model.target = target
        model.opt_level = opt_level
        
        old_dir = os.getcwd()
        os.chdir(model_dir)
        logger.info(f"Change work dir to {model_dir}")
        logger.info("Generate default config file")
        success, msg = gen_cfg(filename, md5_code, target, opt_level, num_cube, toolkit_version, is_transformer)
        if not success:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue

        build_info_json = f"outputs/{target}/result/build_info.json"
        if not os.path.exists(compiled_model_path) or not os.path.exists(build_info_json):
            success, msg = build()
            if not success:
                os.chdir(old_dir)
                logger.error(f"Failed to build: {filepath}")
                model.build_time = build_time
                model.msg = msg
                session.add(model)
                session.commit()
                continue
        if not os.path.exists(build_info_json):
            os.chdir(old_dir)
            msg = "Not found " + build_info_json
            model.msg = msg
            logger.error(msg)
            session.add(model)
            session.commit()
            continue
        model.msg = "ok"
        with open(build_info_json, "r") as f:
            build_info = json.load(f)
        model.build_time = build_info["build_time"]
        prof_json = f"outputs/{target}/result/prof.json"
        if not os.path.exists(prof_json):
            success, msg = run_profile()
            if not success:
                os.chdir(old_dir)
                logger.error(msg)
                model.msg = msg
                session.add(model)
                session.commit()
                continue
        with open(prof_json, "r") as f:
            prof = json.load(f)
        model.compiled_model_path = compiled_model_path
        model.compiled_model_md5 = get_md5_code(compiled_model_path)
        model.build_span = "{:.2f}".format(build_info["build_span"])
        model.quantization_span = "{:.2f}".format(build_info["quantization_span"])
        model.costmodel_latency = "{}".format(prof["cost_model"])
        
        latency, throughput, msg = run_latency(compiled_model_path, 1)
        if msg == "ok":
            model.latency = latency
            model.throughput = throughput
        
        latency, throughput, msg = run_latency(compiled_model_path, 2)
        if msg == "ok":
            model.latency_2t = latency
            model.throughput_2t = throughput

        latency, throughput, msg = run_latency(compiled_model_path, 3)
        if msg == "ok":
            model.latency_3t = latency
            model.throughput_3t = throughput

        latency, throughput, msg = run_latency(compiled_model_path, 4)
        if msg == "ok":
            model.latency_4t = latency
            model.throughput_4t = throughput

        os.chdir(old_dir)
        session.add(model)
        session.commit()
