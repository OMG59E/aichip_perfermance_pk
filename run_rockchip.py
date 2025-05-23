import glob
import subprocess
import os
import sys
import onnx
import numpy as np
import argparse
import json
import shutil
import tarfile
import hashlib
import sqlalchemy
import pymysql
import resource
import traceback
import yaml
import onnx_graphsurgeon as gs
from onnxsim import simplify
from natsort import natsorted
from loguru import logger
from fabric import Connection
from datetime import datetime
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy import Column, Integer, DateTime, String, BOOLEAN, Float, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


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


class Models(Base):
    __tablename__ = "rockchip"
    __table_args__ = {"extend_existing": True, "mysql_charset": "utf8"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    desc = Column(String(512), comment="任务描述")
    model_name = Column(String(512), index=True, comment="模型名")
    md5_code = Column(String(512), comment="模型md5码")
    model_path = Column(String(512), comment="模型路径")
    target = Column(String(512), comment="目标芯片")
    version = Column(String(512), comment="工具链版本")
    msg = Column(LONGTEXT, comment="编译信息")
    build_time = Column(DateTime, comment="编译开始时间")
    build_span = Column(String(512), comment="编译耗时")
    latency_1core = Column(Float, default=0, comment="单核时延, 单线程单核跑1b模型")
    latency = Column(Float, default=0, comment="最低时延, 单线程三核跑1b模型")
    throughput = Column(Float, default=0, comment="最高吞吐, 三线程三核跑1b模型")
    compiled_model_path = Column(String(512), comment="编译后模型路径")
    compiled_model_md5 = Column(String(512), comment="编译后模型MD5")
    

def build_model(target):
    try:
        cmd = f"python -m rknn.api.rknn_convert -t {target} -i config.yml -o ."
        logger.info(cmd)
        build_cmd = ["python", "-m" "rknn.api.rknn_convert", "-t", target, "-i", "config.yml", "-o", "."]
        res = subprocess.run(build_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
        if res.returncode == 0:
            msg = str(res.stdout)
            if msg.find("rknn build fail") != -1:
                return False, msg
            logger.info(f"success:\n{msg}")
            return True, "ok"
        else:
            msg = str(res.stderr)
            logger.error(f"Failed:\n{msg}")
            return False, msg
    except subprocess.TimeoutExpired as e:
        logger.error(f"exception:\n{e.stderr}")
        return False, str(e.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"exception:\n{e.stderr}")
        return False, str(e.stderr)

        
def gen_cfg(filename, md5_code, target, version):
    try:
        cfg_path = "config.yml"
        if not os.path.exists(cfg_path):
            graph = gs.import_onnx(onnx.load(filename))
            input_names = [node.name for node in graph.inputs]
            input_shapes = [list(node.shape) for node in graph.inputs]
            input_dtypes = [str(node.dtype) for node in graph.inputs]
            basename, ext = os.path.splitext(filename)
            config = dict()
            config = {
                "models": {
                    "name": "{}_{}_v{}_{}.rknn".format(basename, target, version, md5_code),
                    "platform": "onnx",
                    "model_file_path": filename,
                    "quantize": True,
                    "dataset": "dataset.txt",
                    "configs": {
                        "mean_values": None,
                        "std_values": None,
                        "quantized_dtype": "asymmetric_quantized-8",
                        "quant_img_RGB2BGR": False,
                        "quantized_algorithm": "normal"
                    }
                },
            }
            with open(cfg_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=True)
        else:
            logger.info(f"Cfg exist, model: {filename}")
        return True, "ok"
    except Exception as e:
        msg = f"Failed to gen config:\n{e}"
        logger.error(msg)
        return False, msg
    
    
def gen_data(filename):
    try:
        # 检查onnx是否仅为batch维度动态
        graph = gs.import_onnx(onnx.load(filename))
        input_names = [node.name for node in graph.inputs]
        input_shapes = [list(node.shape) for node in graph.inputs]
        input_dtypes = [str(node.dtype) for node in graph.inputs]
        npy_list = list()
        for idx, input_name in enumerate(input_names):
            shape = input_shapes[idx]
            shape_name = "x".join(list(map(str, shape)))
            dtype = input_dtypes[idx]
            if dtype not in ["int64", "uint64", "int32", "uint32", "int8", "uint8", "int16", "uint16", "float32", "float64", "float16", "bool"]:
                msg = f"not support input dtype: {dtype}"
                logger.error(msg)
                return False, msg
            bin_name = f"{input_name}_{shape_name}_{dtype}.npy"
            if not os.path.exists(bin_name):
                if dtype in ["int64", "uint64", "int32", "uint32", "int8", "uint8", "int16", "uint16", "bool"]:
                    data = np.random.randint(0, 255, size=shape).astype(dtype=dtype)
                else:
                    data = np.random.uniform(0, 1, size=shape).astype(dtype=dtype)
                np.save(bin_name, data)
            npy_list.append(bin_name)
        with open("dataset.txt", "w") as f:
            f.write(" ".join(npy_list))
        return True, "ok"
    except Exception as e:
        msg = f"Generate random data exception:\n{e}"
        logger.error(msg)
        return False, msg
    
    
def check_onnx_only_batch_dynamic(onnx_path):
    try:
        graph = gs.import_onnx(onnx.load(onnx_path))
        input_names = [node.name for node in graph.inputs]
        input_shapes = [list(node.shape) for node in graph.inputs]
        input_dtypes = [str(node.dtype) for node in graph.inputs]
        _input_shapes = dict()
        for idx, input_name in enumerate(input_names):
            shape = input_shapes[idx]
            if len(shape) >= 2:
                for d in range(1, len(shape)):
                    if not isinstance(shape[d], int) or shape[d] < 1:
                        msg = f"not support dynamic shape: {shape}"
                        logger.error(msg)
                        return {}, msg
            if not isinstance(shape[0], int) or shape[0] < 1:
                shape[0] = 1
                _input_shapes[input_name] = shape
                return _input_shapes, "ok"
        return {}, "ok"
    except Exception as e:
        msg = f"Check onnx exception:\n{e}"
        logger.error(msg)
        return {}, msg
    
    
def get_package_version(package_name):
    import pkg_resources
    try:
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return f"Package {package_name} is not installed."
    
    
def get_md5_code(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def get_latency_throughput(model_name):
    ip_addr = "192.168.33.249"
    username = "radxa"
    password = "radxa"
    remote_work_dir = "/home/radxa/"
    key_filename = "/root/.ssh/id_rsa"
    num_iter = 100
    remote_model_path = os.path.join(remote_work_dir, model_name)
    try:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            logger.info(f"Upload model: {model_name} to {remote_work_dir}")
            c.put(model_name, remote_work_dir)
            run_cmd = f"./rk_run_model {remote_model_path} 0 {num_iter}"
            logger.info(run_cmd)
            # result = c.run(run_cmd, hide=True, timeout=7200, warn=True)
            result = c.run(run_cmd, hide=True, warn=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error:\n{error}"
                logger.error(msg)
                return 0, 0, 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output:\n{output}")
            lines = output.splitlines()
            latency_1core = lines[-1].split(",")[0].split(" ")[-1]
            latency_1core = float(latency_1core[:-2])
            logger.info(f"Exit Code: {result.return_code}")
            
            run_cmd = f"./rk_run_model {remote_model_path} 3 {num_iter}"
            logger.info(run_cmd)
            # result = c.run(run_cmd, hide=True, timeout=7200, warn=True)
            result = c.run(run_cmd, hide=True, warn=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error:\n{error}"
                logger.error(msg)
                return 0, 0, 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output:\n{output}")
            lines = output.splitlines()
            latency = lines[-1].split(",")[0].split(" ")[-1]
            latency = float(latency[:-2])
            logger.info(f"Exit Code: {result.return_code}")

            run_cmd = f"./rk_run_model {remote_model_path} 4 {num_iter}"
            logger.info(run_cmd)
            # result = c.run(run_cmd, hide=True, timeout=7200, warn=True)
            result = c.run(run_cmd, hide=True, warn=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error:\n{error}"
                logger.error(msg)
                return 0, 0, 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output:\n{output}")
            lines = output.splitlines()
            throughput = float(lines[-1].strip().split(" ")[-1])
            logger.info(f"Exit Code: {result.return_code}")

            c.run(f"rm {remote_model_path}", hide=True)
            logger.info(f"Delete model: {remote_model_path}")
            return latency_1core, latency, throughput, "ok"
    except Exception as e:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            c.run(f"rm {remote_model_path}", hide=True)
        msg = f"Run model exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return 0, 0, 0, msg
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rockchip model build tool")
    parser.add_argument("--models", "-m", type=str, required=True, help="Please specify a onnx model dir")
    parser.add_argument("--target", "-t", type=str, required=True, choices=("rk3588", ), help="Please specify a chip target")
    parser.add_argument("--desc", "-d", type=str, required=True, help="Please specify a task desc")
    args = parser.parse_args()
    
    # 检查表是否存在
    tables = engine.execute("show tables;").fetchall()
    tables = [table_name[0] for table_name in tables]
    if Models.__tablename__ not in tables:
        Base.metadata.create_all(engine)
        logger.info("create table success")

    models_dir = args.models
    target = args.target
    desc = args.desc
    toolkit_version = get_package_version("rknn-toolkit2")
    logger.info(f"Toolkit Version: {toolkit_version}")
    work_dir = f"outputs/{target}/{toolkit_version}"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    onnx_files = glob.glob(os.path.join(models_dir, "**/*.onnx"), recursive=True)
    sorted_file_list = natsorted(onnx_files, key=os.path.basename)
    for idx, filepath in enumerate(sorted_file_list):
        if not os.path.exists(filepath):
            logger.info(f"Not found: {filepath}")
            continue
        md5_code = get_md5_code(filepath)
        filename = os.path.basename(filepath)
        basename, ext = os.path.splitext(filename)
        # 检查该模型是不是已经入库
        res = session.query(Models).filter(
            Models.md5_code == md5_code,
            Models.version == toolkit_version,
            Models.target == target,
        ).first()
        if res:
            if res.model_path == filepath:
                logger.info(f"MD5: {md5_code}, Path: {filepath}")
                pass
            else:
                # 路径不一致，直接复制
                model = Models()
                model.desc = desc
                model.model_name = filename
                model.model_path = filepath
                model.md5_code = md5_code
                model.version = toolkit_version
                model.target = target
                model.build_time = res.build_time
                model.build_span = res.build_span
                model.compiled_model_path = res.compiled_model_path
                model.compiled_model_md5 = res.compiled_model_md5
                model.msg = res.msg
                model.latency = res.latency
                model.latency_1core = res.latency_1core
                model.throughput = res.throughput
                session.add(model)
                session.commit()
            continue
        # 新模型
        model = Models()
        model.desc = desc
        model.model_name = filename
        model.model_path = filepath
        model.md5_code = md5_code
        model.version = toolkit_version
        build_time = datetime.now()
        model.build_time = build_time
        model.target = target
        
        model_work_dir = f"{work_dir}/{basename}_{md5_code}/"
        if not os.path.exists(model_work_dir):
            os.makedirs(model_work_dir)
        dst_onnx_path = os.path.join(model_work_dir, filename)
        if not os.path.exists(dst_onnx_path):
            shutil.copy(filepath, model_work_dir)

        old_dir = os.getcwd()
        os.chdir(model_work_dir)
        logger.info(f"Change work dir to {model_work_dir}")
        # 检查是否是动态模型，是且仅batch维度动态，则优化为静态且batch为1
        input_shapes, msg = check_onnx_only_batch_dynamic(filename)
        if 0 == len(input_shapes) and msg != "ok":
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        elif len(input_shapes) > 0:
            new_filename = filename.replace(".onnx", "_b1.onnx")
            new_basename, ext = os.path.splitext(new_filename)
            if not os.path.exists(new_filename):
                # onnxsim 改为静态
                try:
                    model_simp, check = simplify(onnx.load(filename), overwrite_input_shapes=input_shapes)
                except Exception as e:
                    check = False
                if not check:
                    os.chdir(old_dir)
                    msg = f"Failed to onnxsim model: {filename}"
                    model.msg = msg
                    logger.error(msg)
                    session.add(model)
                    session.commit()
                    continue
                onnx.save(model_simp, new_filename)
                logger.info(f"Simplify {filename} to {new_filename}")
            # 更新模型文件名
            filename = new_filename
            basename = new_basename

        logger.info("Generate random data for quantization")
        success, msg = gen_data(filename)       
        if not success:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        success, msg = gen_cfg(filename, md5_code, target, toolkit_version)
        if not success:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        rknn_name = "{}_{}_v{}_{}.rknn".format(basename, target, toolkit_version, md5_code)
        relative_model_path = rknn_name
        if not os.path.exists(relative_model_path):
            success, msg = build_model(target)
            model.build_span = str(int(datetime.timestamp(datetime.now()) - datetime.timestamp(build_time)))
            if not success:
                os.chdir(old_dir)
                logger.error(f"Failed to build: {filepath}")
                model.msg = msg
                session.add(model)
                session.commit()
                continue
        else:
            model.build_span = 0
        # if target == "rk3588":
        #     if rknn_name not in ["ch_PP-OCRv2_rec_infer_b1_rk3588_v2.1.0+708089d1_0ef3605e78c0329c54305f2454e4bcfc.rknn"]:
        #         latency_1core, latency, throughput, msg = get_latency_throughput(rknn_name)
        #         model.latency_1core = latency_1core
        #         model.latency = latency
        #         model.throughput = throughput
        #     else:
        #         logger.error(f"Skip: {rknn_name}")
        #         os.chdir(old_dir)
        #         continue
        # else:
        #     logger.error(f"Not support target: {target}")
        # if latency == 0:
        #     os.chdir(old_dir)
        #     model.msg = msg
        #     session.add(model)
        #     session.commit()
        #     continue
        os.chdir(old_dir)
        compiled_model_path = os.path.join(model_work_dir, relative_model_path)
        model.compiled_model_path = compiled_model_path
        model.compiled_model_md5 = get_md5_code(compiled_model_path)
        model.msg = "ok"
        session.add(model)
        session.commit()
